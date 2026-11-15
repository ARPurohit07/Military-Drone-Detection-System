"""Airframe registry - the single source of truth for the label space.

The point of this file is that no label is ever assigned by hand. Each entry
pairs a specific airframe with the search queries that find pictures of it, so
the scraper searches for an *airframe* ("Bayraktar TB2 drone") and the country
label falls out of the registry rather than out of a human decision.

That inversion is what fixes the old dataset. Previously images were scraped
per country and bulk-filed into Chinese/Turkish/... directories, which is why
60 files ended up byte-identical to images filed under a *different* country: a
country query like "chinese military drone" returns Bayraktars and Herons, and
nothing downstream could tell. An airframe query cannot make that mistake - if
the query was "Shahed-136", the country is Iran by construction.

Three label schemes are derived from the same registry, so changing the
granularity of the task never requires rebuilding the images:

    country     Iran, Turkey, Israel, China, ...    (the original goal)
    airframe    shahed_136, bayraktar_tb2, ...      (Phase 3 platform taxonomy)
    role        male, loitering_munition, ...       (how it is used)

DISTRACTORS holds the not-a-drone classes. They are listed here, and
deliberately numbered after every drone class, so hard negatives can graduate
from background images to predicted classes without renumbering the drone
labels and invalidating every existing checkpoint.
"""

from dataclasses import dataclass


ROLES = {
    "male": "Medium-altitude long-endurance ISR/strike",
    "hale": "High-altitude long-endurance ISR",
    "ucav": "Stealthy / jet-powered unmanned combat air vehicle",
    "loitering_munition": "One-way attack / kamikaze drone",
    "tactical_isr": "Catapult or hand-launched tactical reconnaissance",
    "vtol": "Rotary or vertical-takeoff platform",
}


@dataclass(frozen=True)
class Airframe:
    key: str          # stable identifier, also the image filename prefix
    name: str         # human-readable designation
    country: str      # derived label - never assigned per-image
    role: str         # see ROLES
    queries: tuple    # search strings that return pictures of THIS airframe

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"{self.key}: unknown role {self.role!r}")


def _af(key, name, country, role, *queries):
    return Airframe(key, name, country, role, tuple(queries))


# Queries are deliberately redundant: an airframe has an official designation, a
# popular name and a press name, and image-search coverage differs sharply
# between them ("Shahed-136" and "Geran-2" surface largely disjoint sets).
AIRFRAMES = [
    # ---------------------------------------------------------------- Turkey
    _af("bayraktar_tb2", "Bayraktar TB2", "Turkey", "male",
        "Bayraktar TB2 drone", "Bayraktar TB2 UAV in flight", "TB2 armed drone"),
    _af("bayraktar_akinci", "Bayraktar Akinci", "Turkey", "male",
        "Bayraktar Akinci drone", "Akinci UCAV"),
    _af("bayraktar_tb3", "Bayraktar TB3", "Turkey", "male",
        "Bayraktar TB3 drone"),
    _af("bayraktar_kizilelma", "Bayraktar Kizilelma", "Turkey", "ucav",
        "Bayraktar Kizilelma jet drone", "Kizilelma UCAV"),
    _af("tai_anka", "TAI Anka", "Turkey", "male",
        "TAI Anka drone", "Anka-S UAV Turkish"),
    _af("tai_aksungur", "TAI Aksungur", "Turkey", "male",
        "TAI Aksungur UAV"),
    _af("vestel_karayel", "Vestel Karayel", "Turkey", "tactical_isr",
        "Vestel Karayel UAV"),
    _af("stm_kargu", "STM Kargu", "Turkey", "loitering_munition",
        "STM Kargu drone", "Kargu-2 loitering munition"),

    # ------------------------------------------------------------------ Iran
    _af("shahed_136", "Shahed-136", "Iran", "loitering_munition",
        "Shahed-136 drone", "Geran-2 drone", "Shahed 136 delta wing drone"),
    _af("shahed_129", "Shahed-129", "Iran", "male",
        "Shahed-129 drone", "Shahed 129 UAV Iran"),
    _af("shahed_149", "Shahed-149 Gaza", "Iran", "male",
        "Shahed-149 Gaza drone"),
    _af("mohajer_6", "Mohajer-6", "Iran", "male",
        "Mohajer-6 drone", "Mohajer 6 UAV Iran"),
    _af("mohajer_2", "Mohajer-2", "Iran", "tactical_isr",
        "Mohajer-2 UAV"),
    _af("ababil_3", "Ababil-3", "Iran", "tactical_isr",
        "Ababil-3 UAV Iran", "Ababil 3 drone"),
    _af("karrar", "Karrar", "Iran", "loitering_munition",
        "Karrar drone Iran", "Karrar jet drone"),
    _af("fotros", "Fotros", "Iran", "male",
        "Fotros UAV Iran"),
    _af("kaman_22", "Kaman-22", "Iran", "male",
        "Kaman-22 drone Iran"),

    # ---------------------------------------------------------------- Israel
    _af("iai_heron", "IAI Heron", "Israel", "male",
        "IAI Heron drone", "Heron UAV Israel"),
    _af("iai_heron_tp", "IAI Heron TP", "Israel", "male",
        "Heron TP Eitan drone"),
    _af("hermes_900", "Elbit Hermes 900", "Israel", "male",
        "Elbit Hermes 900 drone", "Hermes 900 UAV"),
    _af("hermes_450", "Elbit Hermes 450", "Israel", "male",
        "Elbit Hermes 450 UAV"),
    _af("iai_harop", "IAI Harop", "Israel", "loitering_munition",
        "IAI Harop loitering munition", "Harop drone"),
    _af("iai_harpy", "IAI Harpy", "Israel", "loitering_munition",
        "IAI Harpy drone"),
    _af("iai_searcher", "IAI Searcher", "Israel", "tactical_isr",
        "IAI Searcher UAV"),
    _af("orbiter_3", "Aeronautics Orbiter 3", "Israel", "tactical_isr",
        "Aeronautics Orbiter 3 UAV"),

    # ----------------------------------------------------------------- China
    _af("wing_loong_1", "Wing Loong I", "China", "male",
        "Wing Loong I drone", "Pterodactyl I UAV China"),
    _af("wing_loong_2", "Wing Loong II", "China", "male",
        "Wing Loong II drone", "Wing Loong 2 UAV"),
    _af("ch_4", "CASC CH-4", "China", "male",
        "CH-4 Rainbow drone China", "Caihong 4 UAV"),
    _af("ch_5", "CASC CH-5", "China", "male",
        "CH-5 Rainbow drone China"),
    _af("gj_11", "GJ-11 Sharp Sword", "China", "ucav",
        "GJ-11 Sharp Sword stealth drone", "Hongdu GJ-11 UCAV"),
    _af("wz_7", "WZ-7 Soaring Dragon", "China", "hale",
        "WZ-7 Soaring Dragon UAV", "Xianglong drone China"),
    _af("wz_8", "WZ-8", "China", "hale",
        "WZ-8 supersonic reconnaissance drone"),
    _af("asn_209", "ASN-209", "China", "tactical_isr",
        "ASN-209 UAV China"),

    # ------------------------------------------------------------------- USA
    _af("mq9_reaper", "General Atomics MQ-9 Reaper", "USA", "male",
        "MQ-9 Reaper drone", "MQ-9 Reaper in flight"),
    _af("mq1_predator", "General Atomics MQ-1 Predator", "USA", "male",
        "MQ-1 Predator drone", "Predator UAV US Air Force"),
    _af("rq4_global_hawk", "Northrop RQ-4 Global Hawk", "USA", "hale",
        "RQ-4 Global Hawk drone", "Global Hawk UAV"),
    _af("rq170_sentinel", "Lockheed RQ-170 Sentinel", "USA", "ucav",
        "RQ-170 Sentinel stealth drone"),
    _af("mq1c_gray_eagle", "MQ-1C Gray Eagle", "USA", "male",
        "MQ-1C Gray Eagle drone"),
    _af("rq7_shadow", "AAI RQ-7 Shadow", "USA", "tactical_isr",
        "RQ-7 Shadow UAV"),
    _af("switchblade", "AeroVironment Switchblade", "USA", "loitering_munition",
        "AeroVironment Switchblade 300 drone"),
    _af("rq11_raven", "AeroVironment RQ-11 Raven", "USA", "tactical_isr",
        "RQ-11 Raven UAV soldier"),
    _af("x47b", "Northrop X-47B", "USA", "ucav",
        "X-47B unmanned combat air vehicle"),

    # ---------------------------------------------------------------- Russia
    _af("orlan_10", "Orlan-10", "Russia", "tactical_isr",
        "Orlan-10 drone Russia", "Orlan 10 UAV"),
    _af("lancet_3", "ZALA Lancet-3", "Russia", "loitering_munition",
        "ZALA Lancet-3 drone", "Lancet loitering munition"),
    _af("orion_inokhodets", "Kronshtadt Orion", "Russia", "male",
        "Kronshtadt Orion drone", "Inokhodets UAV Russia"),
    _af("forpost", "Forpost", "Russia", "male",
        "Forpost UAV Russia"),
    _af("s70_okhotnik", "Sukhoi S-70 Okhotnik", "Russia", "ucav",
        "Sukhoi S-70 Okhotnik drone"),
    _af("eleron_3", "Eleron-3", "Russia", "tactical_isr",
        "Eleron-3SV UAV Russia"),

    # ----------------------------------------------------------------- India
    _af("tapas_bh", "DRDO TAPAS-BH Rustom-II", "India", "male",
        "DRDO TAPAS-BH drone", "Rustom-2 UAV India"),
    _af("drdo_nishant", "DRDO Nishant", "India", "tactical_isr",
        "DRDO Nishant UAV"),
    _af("drishti_10", "Adani Drishti 10 Starliner", "India", "male",
        "Drishti 10 Starliner drone"),
    _af("drdo_ghatak", "DRDO Ghatak", "India", "ucav",
        "DRDO Ghatak stealth UCAV"),

    # -------------------------------------------------------------- Pakistan
    _af("burraq", "NESCOM Burraq", "Pakistan", "male",
        "NESCOM Burraq drone Pakistan"),
    _af("shahpar_2", "GIDS Shahpar-II", "Pakistan", "male",
        "Shahpar-II UAV Pakistan"),

    # --------------------------------------------------------------- Ukraine
    _af("uj22_airborne", "UkrJet UJ-22 Airborne", "Ukraine", "loitering_munition",
        "UJ-22 Airborne drone Ukraine"),
    _af("leleka_100", "Leleka-100", "Ukraine", "tactical_isr",
        "Leleka-100 UAV Ukraine"),
    _af("pd_2", "Ukrspecsystems PD-2", "Ukraine", "male",
        "Ukrspecsystems PD-2 UAV"),

    # ---------------------------------------------------------------- Europe
    _af("watchkeeper", "Thales Watchkeeper WK450", "UK", "tactical_isr",
        "Thales Watchkeeper WK450 drone"),
    _af("taranis", "BAE Taranis", "UK", "ucav",
        "BAE Systems Taranis UCAV"),
    _af("neuron", "Dassault nEUROn", "France", "ucav",
        "Dassault nEUROn stealth drone"),
    _af("safran_patroller", "Safran Patroller", "France", "male",
        "Safran Patroller UAV"),
    _af("piaggio_p1hh", "Piaggio P.1HH HammerHead", "Italy", "male",
        "Piaggio P.1HH HammerHead UAV", "P.1HH HammerHead drone"),
    _af("luna_ng", "Rheinmetall LUNA NG", "Germany", "tactical_isr",
        "Rheinmetall LUNA NG UAV"),

    # ----------------------------------------------------------- South Korea
    _af("kus_fs", "KAI KUS-FS", "South Korea", "male",
        "KAI KUS-FS drone Korea", "KUS-FS UAV"),

    # ======================================================================
    # Depth pass. The registry above averages ~4.5 airframes per country, but
    # unevenly: the USA and Iran had 9 while Italy, Germany and South Korea had
    # one each. A country with one airframe collects ~100 images, which is far
    # too few to learn a class from and would overfit regardless of how large the
    # dataset is overall - the imbalance is the risk, not the total.
    #
    # These entries deepen the countries already present rather than adding new
    # ones, because every new country would arrive as another thin class. The
    # weighting is deliberately toward the shallow end of the list.
    # ======================================================================

    # ---------------------------------------------------------------- Turkey
    _af("tai_anka_3", "TAI Anka-3", "Turkey", "ucav",
        "TAI Anka-3 flying wing drone", "Anka-3 UCAV Turkey"),
    _af("stm_alpagut", "STM Alpagut", "Turkey", "loitering_munition",
        "STM Alpagut loitering munition"),
    _af("baykar_mini", "Bayraktar Mini UAV", "Turkey", "tactical_isr",
        "Bayraktar Mini UAV hand launched"),
    _af("asisguard_songar", "ASISGUARD Songar", "Turkey", "vtol",
        "ASISGUARD Songar armed drone"),

    # ------------------------------------------------------------------ Iran
    _af("shahed_191", "Shahed-191 Saegheh", "Iran", "ucav",
        "Shahed-191 Saegheh drone", "Saegheh flying wing drone Iran"),
    _af("shahed_171", "Shahed-171 Simorgh", "Iran", "ucav",
        "Shahed-171 Simorgh drone"),
    _af("arash_2", "Arash-2", "Iran", "loitering_munition",
        "Arash-2 loitering munition Iran"),
    _af("meraj_521", "Meraj-521", "Iran", "loitering_munition",
        "Meraj-521 drone Iran"),
    _af("yasir", "Yasir", "Iran", "tactical_isr",
        "Yasir UAV Iran"),

    # ---------------------------------------------------------------- Israel
    _af("elbit_skylark", "Elbit Skylark", "Israel", "tactical_isr",
        "Elbit Skylark UAV", "Skylark I-LEX drone"),
    _af("iai_bird_eye", "IAI Bird-Eye", "Israel", "tactical_isr",
        "IAI Bird-Eye 400 UAV"),
    _af("iai_panther", "IAI Panther", "Israel", "vtol",
        "IAI Panther tilt rotor UAV"),
    _af("aeronautics_dominator", "Aeronautics Dominator", "Israel", "male",
        "Aeronautics Dominator XP UAV"),

    # ----------------------------------------------------------------- China
    _af("ch_3", "CASC CH-3", "China", "male",
        "CH-3 Rainbow UAV China"),
    _af("ch_7", "CASC CH-7", "China", "ucav",
        "CH-7 Rainbow stealth drone", "Caihong 7 UCAV"),
    _af("wing_loong_3", "Wing Loong III", "China", "male",
        "Wing Loong 3 drone China"),
    _af("cloud_shadow", "Cloud Shadow", "China", "male",
        "Cloud Shadow UAV China", "Yunying drone"),
    _af("tb_001", "TB-001 Scorpion", "China", "male",
        "TB-001 Scorpion drone China"),
    _af("wj_700", "WJ-700 Falcon Hunter", "China", "male",
        "WJ-700 Falcon Hunter drone"),
    _af("bzk_005", "Harbin BZK-005", "China", "male",
        "BZK-005 UAV China"),

    # ------------------------------------------------------------------- USA
    _af("mq4c_triton", "Northrop MQ-4C Triton", "USA", "hale",
        "MQ-4C Triton drone"),
    _af("mq8_fire_scout", "Northrop MQ-8 Fire Scout", "USA", "vtol",
        "MQ-8 Fire Scout helicopter drone"),
    _af("mq20_avenger", "General Atomics MQ-20 Avenger", "USA", "ucav",
        "MQ-20 Avenger drone", "Predator C Avenger"),
    _af("xq58_valkyrie", "Kratos XQ-58 Valkyrie", "USA", "ucav",
        "XQ-58 Valkyrie drone"),
    _af("rq21_blackjack", "Insitu RQ-21 Blackjack", "USA", "tactical_isr",
        "RQ-21 Blackjack UAV"),
    _af("scaneagle", "Insitu ScanEagle", "USA", "tactical_isr",
        "ScanEagle UAV catapult launch"),
    _af("mq25_stingray", "Boeing MQ-25 Stingray", "USA", "ucav",
        "MQ-25 Stingray drone"),

    # ---------------------------------------------------------------- Russia
    _af("kub_bla", "ZALA KUB-BLA", "Russia", "loitering_munition",
        "ZALA KUB-BLA loitering munition"),
    _af("altius_u", "Altius-U", "Russia", "male",
        "Altius-U drone Russia", "Altair UAV Russia"),
    _af("orlan_30", "Orlan-30", "Russia", "tactical_isr",
        "Orlan-30 UAV Russia"),
    _af("korsar", "Korsar", "Russia", "tactical_isr",
        "Korsar UAV Russia"),
    _af("sirius_uav", "Kronshtadt Sirius", "Russia", "male",
        "Kronshtadt Sirius UAV"),

    # ----------------------------------------------------------------- India
    _af("nagastra_1", "Nagastra-1", "India", "loitering_munition",
        "Nagastra-1 loitering munition India"),
    _af("archer_ng", "DRDO Archer-NG", "India", "male",
        "DRDO Archer-NG drone"),
    _af("rustom_1", "DRDO Rustom-1", "India", "male",
        "DRDO Rustom-1 UAV"),

    # -------------------------------------------------------------- Pakistan
    _af("nescom_uqab", "NESCOM Uqab", "Pakistan", "tactical_isr",
        "NESCOM Uqab UAV Pakistan"),
    _af("shahpar_3", "GIDS Shahpar-III", "Pakistan", "male",
        "Shahpar-III UAV Pakistan"),

    # --------------------------------------------------------------- Ukraine
    _af("bober", "Bober", "Ukraine", "loitering_munition",
        "Bober drone Ukraine"),
    _af("ram_ii", "RAM II", "Ukraine", "loitering_munition",
        "RAM II loitering munition Ukraine"),
    _af("punisher_uav", "Punisher", "Ukraine", "loitering_munition",
        "Punisher drone Ukraine"),
    _af("furia", "Furia", "Ukraine", "tactical_isr",
        "Furia UAV Ukraine"),

    # -------------------------------------------------------------------- UK
    _af("protector_rg1", "RAF Protector RG Mk1", "UK", "male",
        "Protector RG Mk1 drone RAF", "MQ-9B SkyGuardian"),
    _af("banshee", "QinetiQ Banshee", "UK", "tactical_isr",
        "QinetiQ Banshee target drone"),
    _af("malloy_t150", "Malloy T-150", "UK", "vtol",
        "Malloy T-150 heavy lift drone"),

    # ---------------------------------------------------------------- France
    _af("sagem_sperwer", "Sagem Sperwer", "France", "tactical_isr",
        "Sagem Sperwer UAV"),
    _af("aliaca", "Survey Copter Aliaca", "France", "tactical_isr",
        "Aliaca UAV French navy"),
    _af("delair_dt26", "Delair DT26", "France", "tactical_isr",
        "Delair DT26 UAV"),

    # ----------------------------------------------------------------- Italy
    _af("falco_evo", "Leonardo Falco EVO", "Italy", "male",
        "Leonardo Falco EVO UAV", "Falco EVO drone"),
    _af("falco_xplorer", "Leonardo Falco Xplorer", "Italy", "male",
        "Leonardo Falco Xplorer drone", "Falco Xplorer UAV"),
    _af("awhero", "Leonardo AWHero", "Italy", "vtol",
        "Leonardo AWHero rotary UAV", "AWHero drone"),

    # --------------------------------------------------------------- Germany
    _af("quantum_vector", "Quantum Systems Vector", "Germany", "tactical_isr",
        "Quantum Systems Vector UAV"),
    _af("emt_aladin", "EMT Aladin", "Germany", "tactical_isr",
        "EMT Aladin UAV Bundeswehr"),
    _af("do_dt25", "Airbus Do-DT25", "Germany", "tactical_isr",
        "Do-DT25 target drone Airbus"),

    # ----------------------------------------------------------- South Korea
    _af("night_intruder", "KAI RQ-101 Night Intruder", "South Korea",
        "tactical_isr", "RQ-101 Night Intruder UAV Korea", "RQ-101 UAV"),
    _af("kus_vs", "Korean Air KUS-VS", "South Korea", "vtol",
        "Korean Air KUS-VS tiltrotor drone", "KUS-VS UAV"),
    _af("devil_killer", "Devil Killer", "South Korea", "loitering_munition",
        "Devil Killer loitering munition Korea", "KAI Devil Killer UAV"),

    # ---------------------------------------------------------------- Poland
    # WB Group publishes these as distinct military UAS families. Keeping the
    # platform name in every query is important: broad "Polish drone" searches
    # return imported Bayraktars and would recreate the old label-noise problem.
    _af("wb_flyeye", "WB Electronics FlyEye", "Poland", "tactical_isr",
        "WB FlyEye military UAV", "FlyEye mini UAS Poland"),
    _af("wb_ft5", "WB Group FT-5 Los", "Poland", "tactical_isr",
        "WB Group FT-5 Los UAV", "FT-5 tactical aerial platform Poland"),
    _af("warmate_3", "WB Group Warmate 3", "Poland", "loitering_munition",
        "Warmate 3 loitering munition", "WB Warmate drone Poland"),
    _af("warmate_5", "WB Group Warmate 5", "Poland", "loitering_munition",
        "Warmate 5 loitering munition", "WB Group Warmate 5 UAV"),

    # --------------------------------------------------- United Arab Emirates
    _af("reach_s", "EDGE REACH-S", "UAE", "male",
        "EDGE REACH-S UAV", "REACH-S MALE drone UAE", "REACH-S drone"),
    _af("garmoosha", "ADASI Garmoosha", "UAE", "vtol",
        "ADASI Garmoosha UAV", "Garmoosha rotary wing drone UAE",
        "Garmoosha drone"),
    _af("jeniah", "EDGE Jeniah", "UAE", "ucav",
        "EDGE Jeniah UCAV", "Jeniah unmanned combat aircraft UAE",
        "Jeniah drone"),
    _af("qx_6_50", "EDGE QX-6 50", "UAE", "loitering_munition",
        "EDGE QX-6 50 loitering munition", "QX 6-50 drone UAE",
        "QX-6 50 drone"),
    _af("shadow_50", "EDGE SHADOW 50", "UAE", "loitering_munition",
        "EDGE SHADOW 50 loitering munition", "SHADOW 50 drone UAE",
        "EDGE SHADOW 50 drone"),

    # ---------------------------------------------------------------- Serbia
    _af("pegaz_011", "Pegaz 011", "Serbia", "tactical_isr",
        "Pegaz 011 UAV Serbia", "Pegasus Serbian military drone"),
    _af("gavran_145", "Gavran 145", "Serbia", "loitering_munition",
        "Gavran 145 UAV Serbia", "Gavran Raven loitering munition Serbia"),
    _af("vrabac", "Vrabac", "Serbia", "tactical_isr",
        "Vrabac UAV Serbia", "Vrabac Sparrow military drone"),
    _af("strsljen", "Strsljen", "Serbia", "vtol",
        "Strsljen unmanned helicopter Serbia", "Strsljen UAV Yugoimport"),

    # ---------------------------------------------------------- South Africa
    _af("milkor_380", "Milkor 380", "South Africa", "male",
        "Milkor 380 UAV", "Milkor 380 UCAV South Africa"),
    _af("denel_seeker_2", "Denel Seeker II", "South Africa", "tactical_isr",
        "Denel Seeker II UAV", "Seeker 2 drone South Africa"),
    _af("denel_seeker_400", "Denel Seeker 400", "South Africa", "male",
        "Denel Seeker 400 UAV", "Seeker 400 drone South Africa"),
    _af("n_raven", "Paramount N-Raven", "South Africa", "loitering_munition",
        "Paramount N-Raven loitering munition", "N-Raven UAV"),
]


# Image search returns a largely fixed set per query string, so the cheapest way
# to raise unique yield per airframe is to ask differently rather than to ask
# again. These suffixes are appended to each airframe's human-readable
# designation, which is distinctive enough ("MQ-9 Reaper takeoff") that the
# results stay on-subject.
#
# This is safe against mislabelling in a way a broader query would not be: the
# country still comes from the registry entry, never from the result. A variant
# that returns an unrelated aircraft costs one image that the auto-labeller will
# usually reject anyway - it cannot silently relabel a class.
QUERY_VARIANTS = (
    "{} in flight",
    "{} drone photo",
    "{} takeoff",
    "{} military UAV",
)


def all_queries(airframe):
    """Explicit queries for an airframe, plus generated variants.

    Order matters: the hand-written queries run first because they carry the
    distinctive aliases (Shahed-136 vs Geran-2) that surface disjoint image sets.
    Variants are a tail used only when an airframe has not hit its target.
    """
    seen = list(airframe.queries)
    lowered = {q.lower() for q in seen}
    for template in QUERY_VARIANTS:
        candidate = template.format(airframe.name)
        if candidate.lower() not in lowered:
            seen.append(candidate)
            lowered.add(candidate.lower())
    return tuple(seen)


# Not-a-drone classes. Numbered AFTER every drone class (see build_label_map) so
# that promoting these from background images to predicted classes does not
# shift a single drone label, and existing checkpoints stay loadable.
#
# The values match category prefixes already present in the Negative image pool,
# so the 380 negatives that were already scraped map straight onto them.
DISTRACTORS = {
    "bird": ("birds_in_sky", "seagulls", "bats_flying"),
    "airliner": ("planes_far",),
    "helicopter": ("helicopters_far",),
    "balloon": ("balloons_in_sky",),
    "kite": ("kites_flying",),
    "insect": ("insects_flying",),
    # An RC aircraft is the single most drone-like thing in the sky that is not a
    # military drone, and a paraglider has the wing-plus-payload silhouette that
    # a fixed-wing detector latches onto. Both earn their own class rather than
    # being folded into "airliner".
    "model_aircraft": ("model_aircraft",),
    "paraglider": ("paragliders",),
}

# Negative categories with no plausible drone-like object in them at all. These
# are pure background: useful for suppressing false positives, but there is
# nothing in them to ever draw a box around.
PURE_BACKGROUND = (
    "sky", "sky_clouds", "landscape", "trees", "branches_sky", "leaves_in_sky",
    "curtains", "fingers", "hands", "people", "room", "windows",
    "floating_plastic_bag",
    # Cluttered skyline and night-sky categories. Nothing in them is drone-shaped,
    # but they are where a detector looking at real sky produces false positives:
    # a streetlight or antenna against bright cloud reads as a small dark object.
    "night_sky", "streetlights", "power_lines", "antennas", "wind_turbines",
    "fireworks",
    # Deployment-specific hard negatives found by the camera stress test. The
    # system is intended to detect a physical airborne object, so a drone photo
    # on a phone/monitor is background even though its pixels contain a drone.
    "drone_on_screen", "phone_closeup", "indoor_person_closeup",
    "deployment_false_positive",
)

BY_KEY = {a.key: a for a in AIRFRAMES}
COUNTRIES = sorted({a.country for a in AIRFRAMES})
ROLE_KEYS = sorted({a.role for a in AIRFRAMES})

if len(BY_KEY) != len(AIRFRAMES):
    raise RuntimeError("duplicate airframe key in registry")


def build_label_map(scheme="country", include_distractors=False):
    """Return {class_name: label} with foreground labels starting at 1.

    Label 0 is torchvision's background class and is never handed out here.
    Getting this wrong is what made "Turkish Drones" score exactly 0.000 AP in
    the Phase 1 baseline: it held label 0, so every one of its boxes was fed to
    the model as background and the class was untrainable.
    """
    if scheme == "country":
        names = COUNTRIES
    elif scheme == "airframe":
        names = sorted(BY_KEY)
    elif scheme == "role":
        names = ROLE_KEYS
    else:
        raise ValueError(f"unknown scheme {scheme!r}; "
                         "expected 'country', 'airframe' or 'role'")

    label_map = {n: i + 1 for i, n in enumerate(names)}
    if include_distractors:
        base = len(label_map)
        for i, d in enumerate(sorted(DISTRACTORS)):
            label_map[d] = base + i + 1
    return label_map


def class_for(key, scheme="country"):
    """Map an airframe key to its class name under `scheme`."""
    af = BY_KEY[key]
    return {"country": af.country, "airframe": af.key, "role": af.role}[scheme]


def summary():
    from collections import Counter
    return {
        "airframes": len(AIRFRAMES),
        "countries": len(COUNTRIES),
        "queries": sum(len(a.queries) for a in AIRFRAMES),
        "queries_with_variants": sum(len(all_queries(a)) for a in AIRFRAMES),
        "by_country": dict(Counter(a.country for a in AIRFRAMES).most_common()),
        "by_role": dict(Counter(a.role for a in AIRFRAMES).most_common()),
    }


def projected_yield(per_query=55, target_per_airframe=110):
    """Rough projected image count, to sanity-check the registry's capacity.

    `per_query` is the observed number of unique keepers a single image-search
    query yields after duplicate and dead-link rejection (measured at ~55-85).
    Capped per airframe, because extra queries mostly resurface the same photos.
    """
    total = 0
    per_country = {}
    for af in AIRFRAMES:
        got = min(target_per_airframe, per_query * len(all_queries(af)))
        total += got
        per_country[af.country] = per_country.get(af.country, 0) + got
    return total, dict(sorted(per_country.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    import json
    s = summary()
    print(f"{s['airframes']} airframes, {s['countries']} countries, "
          f"{s['queries']} explicit queries "
          f"({s['queries_with_variants']} with variants)")
    print("\nairframes per country:")
    for k, v in s["by_country"].items():
        flag = "   <- thin" if v < 3 else ""
        print(f"  {v:3d}  {k}{flag}")
    print("\nby role:")
    for k, v in s["by_role"].items():
        print(f"  {v:3d}  {k}")

    total, per_country = projected_yield()
    print(f"\nprojected drone images: ~{total}")
    print("projected per country (the class balance that matters):")
    for k, v in per_country.items():
        print(f"  {v:5d}  {k}")

    print("\ncountry label map (with distractors):")
    print(json.dumps(build_label_map("country", include_distractors=True), indent=2))
