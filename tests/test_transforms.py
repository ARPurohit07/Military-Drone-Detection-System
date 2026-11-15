import unittest

import torch

from data.transforms import ResizeLongestSide, get_val_transforms


class ResizeLongestSideTests(unittest.TestCase):
    def test_downscales_image_and_boxes_together(self):
        image = torch.zeros((3, 40, 80), dtype=torch.float32)
        target = {"boxes": torch.tensor([[8.0, 4.0, 72.0, 36.0]])}

        resized, target = ResizeLongestSide(20)(image, target)

        self.assertEqual(tuple(resized.shape), (3, 10, 20))
        torch.testing.assert_close(
            target["boxes"], torch.tensor([[2.0, 1.0, 18.0, 9.0]]))

    def test_does_not_upscale_small_images(self):
        image = torch.zeros((3, 10, 20), dtype=torch.float32)
        boxes = torch.tensor([[2.0, 1.0, 18.0, 9.0]])
        resized, target = ResizeLongestSide(80)(image, {"boxes": boxes})
        self.assertIs(resized, image)
        self.assertIs(target["boxes"], boxes)

    def test_validation_caps_large_inputs_and_scales_boxes(self):
        image = torch.zeros((3, 1000, 2000), dtype=torch.float32)
        target = {"boxes": torch.tensor([[200.0, 100.0, 1800.0, 900.0]])}

        resized, target = get_val_transforms()(image, target)

        self.assertEqual(tuple(resized.shape), (3, 400, 800))
        torch.testing.assert_close(
            target["boxes"], torch.tensor([[80.0, 40.0, 720.0, 360.0]]))


if __name__ == "__main__":
    unittest.main()
