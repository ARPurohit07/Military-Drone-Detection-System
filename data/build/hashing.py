"""Perceptual hashing for near-duplicate detection.

The old dataset's core defect was undetected duplication: 1325 annotated images
held only 1292 unique pictures, 21 near-duplicate pairs straddled the train/val
boundary (so val was partly memorised), and 39 pairs showed the *same aircraft
labelled as two different countries*. Exact byte hashing misses all of the
interesting cases - a re-encoded, resized or lightly cropped copy of an image
has a completely different md5 - so every stage of the rebuild dedups on a
perceptual hash instead.

dhash (difference hash) is used rather than an average hash because it keys on
local gradient direction, which survives JPEG re-encoding and global brightness
shifts. That matters here: the same press photo of a Bayraktar is republished at
a dozen sizes and compression levels.

`imagehash` is not a dependency of this project, so the 64-bit dhash is
implemented directly against numpy - it is nine lines and avoids adding a
package for it.
"""

import numpy as np
import cv2


HASH_BITS = 64
# Two images within this Hamming distance are treated as the same picture.
# Chosen from the measured distribution on the existing corpus: byte-identical
# re-encodes land at 0-2, genuine crops and rescales at 3-6, and distinct photos
# of the same airframe sit well above 12. 6 keeps re-encodes and crops while not
# collapsing different photos of one aircraft.
DUP_THRESHOLD = 6

# A dhash is only meaningful when the image actually has gradient structure. A
# near-uniform picture - blank sky, a white studio background, a silhouette -
# produces a hash with almost no bits set, and two such hashes can sit within
# DUP_THRESHOLD of each other purely because neither has any bits to differ in.
#
# This is not hypothetical. On the first full build it merged 15 unrelated images
# - a Bayraktar TB2, an RQ-170, a Taranis, an nEUROn, four "fingers" negatives and
# a bird - into a single "conflict" group. Every member had 4-12 of 64 bits set
# against a healthy ~32, and most pairs inside the group were 7-14 apart, i.e.
# NOT duplicates; a couple of accidental sub-threshold links chained the rest
# together.
#
# Hashes outside this band are treated as uninformative and excluded from
# clustering: the image is kept as its own picture rather than matched on
# evidence that does not exist.
MIN_HASH_BITS = 10
MAX_HASH_BITS = HASH_BITS - MIN_HASH_BITS

# Rows per chunk when computing pairwise distances. The naive
# `bits[:, None, :] != bits[None, :, :]` allocates an (N, N, 64) bool array,
# which is 8.4 GB at N=11461 - enough to push this machine into a
# low-memory condition mid-build. Chunking bounds it to CHUNK x N x 64.
CHUNK_ROWS = 512


def dhash(image, hash_size=8):
    """Return a 64-bit difference hash as a uint8 array of length 8.

    `image` is a BGR or grayscale numpy array. Returns None if unreadable.
    """
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hash_size+1 columns so that adjacent-column comparison yields hash_size
    # bits per row.
    small = cv2.resize(image, (hash_size + 1, hash_size),
                       interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff.flatten())


def dhash_file(path):
    """dhash an image file, or None if it cannot be decoded."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return dhash(img)


def dhash_bytes(raw):
    """dhash raw encoded image bytes, without touching disk."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return dhash(img)


def hamming(a, b):
    """Hamming distance between two packed hashes."""
    return int(np.unpackbits(a ^ b).sum())


def hash_bit_count(h):
    """Number of set bits - a proxy for how much gradient structure the image has."""
    return int(np.unpackbits(np.asarray(h, dtype=np.uint8)).sum())


def is_informative(h, min_bits=MIN_HASH_BITS, max_bits=MAX_HASH_BITS):
    """Whether a hash carries enough structure for distance to mean anything."""
    if h is None:
        return False
    return min_bits <= hash_bit_count(h) <= max_bits


def hamming_matrix(hashes, chunk_rows=CHUNK_ROWS):
    """Pairwise Hamming distances for a stack of packed hashes.

    `hashes` is an (N, 8) uint8 array. Returns an (N, N) uint8 array with the
    diagonal set to HASH_BITS + 1 so self-matches never count as duplicates.

    Computed in row chunks: the one-shot broadcast form needs (N, N, 64) bytes,
    which is 8.4 GB at N~11.5k. Output is uint8 because distances cannot exceed
    65, which keeps the result itself to N^2 bytes rather than N^2 * 8.
    """
    bits = np.unpackbits(np.asarray(hashes, dtype=np.uint8), axis=1)
    n = len(bits)
    dist = np.empty((n, n), dtype=np.uint8)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = (bits[start:stop, None, :] != bits[None, :, :]).sum(axis=2)
        dist[start:stop] = block.astype(np.uint8)
    np.fill_diagonal(dist, HASH_BITS + 1)
    return dist


def candidate_pairs(hashes, threshold=DUP_THRESHOLD, chunk_rows=CHUNK_ROWS,
                    skip=None):
    """Yield (i, j) index pairs within `threshold`, i < j.

    Streams in row chunks so nothing larger than chunk_rows x N is held at once,
    and never materialises the full distance matrix.

    `skip` is a boolean mask of entries to exclude from matching entirely - used
    for uninformative hashes, which would otherwise match each other on the
    absence of structure rather than on content.
    """
    bits = np.unpackbits(np.asarray(hashes, dtype=np.uint8), axis=1)
    n = len(bits)
    if skip is None:
        skip = np.zeros(n, dtype=bool)
    skip = np.asarray(skip, dtype=bool)

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = (bits[start:stop, None, :] != bits[None, :, :]).sum(axis=2)
        rows, cols = np.where(block <= threshold)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if j <= i:
                continue          # upper triangle only, and no self-match
            if skip[i] or skip[j]:
                continue
            yield i, j


class DuplicateIndex:
    """Incremental near-duplicate filter.

    Used by the scraper to reject a download that merely repeats something
    already kept, without holding the images in memory. Linear scan per query is
    fine at this corpus size (tens of thousands of hashes); a BK-tree would only
    pay off an order of magnitude higher.
    """

    def __init__(self, threshold=DUP_THRESHOLD, require_informative=True):
        self.threshold = threshold
        # Without this guard the scraper rejects blank-sky and white-background
        # photographs as duplicates of each other, because their hashes have
        # almost no bits set and so sit within `threshold` by default. Those are
        # legitimate images and there is no second chance to recover them: the
        # download is discarded on the spot.
        self.require_informative = require_informative
        self._hashes = []
        self._keys = []

    def __len__(self):
        return len(self._hashes)

    def find(self, h):
        """Return the key of a near-duplicate already in the index, else None."""
        if h is None or not self._hashes:
            return None
        if self.require_informative and not is_informative(h):
            return None
        stack = np.asarray(self._hashes, dtype=np.uint8)
        bits = np.unpackbits(stack ^ h[None, :], axis=1).sum(axis=1)
        i = int(bits.argmin())
        return self._keys[i] if bits[i] <= self.threshold else None

    def add(self, h, key):
        if h is not None:
            self._hashes.append(h)
            self._keys.append(key)

    def add_if_new(self, h, key):
        """Add `h` unless a near-duplicate exists. Returns the clashing key."""
        hit = self.find(h)
        if hit is None:
            self.add(h, key)
        return hit


def ncc(image_a, image_b, size=64):
    """Normalised cross-correlation of two grayscale images, in [-1, 1].

    Resizes both to a common size and zero-means them, so the result is invariant
    to scale and to brightness/contrast shifts. Byte-identical or merely
    re-encoded pictures score ~1.0; unrelated pictures score near 0. This is the
    same check that established the legacy dataset's mislabelled pairs were
    genuinely the same photo (NCC = 1.000), rather than a hash coincidence.
    """
    if image_a is None or image_b is None:
        return 0.0
    a = cv2.resize(image_a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(image_b, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    da, db = a.std(), b.std()
    if da < 1e-6 or db < 1e-6:
        # One of them is genuinely flat; correlation is undefined, so do not
        # claim a match.
        return 0.0
    return float((a * b).mean() / (da * db))


def group_duplicates(hashes, threshold=DUP_THRESHOLD, verify=None,
                     require_informative=True, chunk_rows=CHUNK_ROWS):
    """Cluster indices whose hashes are within `threshold` of each other.

    Single-link clustering via union-find, so a chain of near-duplicates
    (A~B, B~C) lands in one group even when A and C are further apart. That is
    the behaviour wanted for splitting: every copy of a picture must end up on
    the same side of the train/val boundary.

    Chaining is also what makes a single bad link expensive, because it merges two
    otherwise unrelated clusters. Two guards apply:

    `require_informative` drops hashes with too little structure for distance to
    be meaningful (see MIN_HASH_BITS) - those images are returned as singletons.

    `verify(i, j) -> bool` is an optional second opinion consulted for every
    candidate pair before it is allowed to union. Only verified links form
    clusters, so one accidental sub-threshold match can no longer drag fifteen
    unrelated pictures into one group. Pass a content check such as `ncc`.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if n:
        skip = None
        if require_informative:
            skip = np.array([not is_informative(h) for h in hashes], dtype=bool)
        for i, j in candidate_pairs(hashes, threshold, chunk_rows, skip):
            if verify is not None and not verify(i, j):
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
