from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
import json

def collate_img(batch_data):
    img_path_batch_list = list()
    name_batch_list = list()
    init_caption_list = list()
    for unit in batch_data:
        img_path_batch_list.append(unit[0])
        name_batch_list.append(unit[1])
        init_caption_list.append(unit[2])
    return img_path_batch_list,name_batch_list, init_caption_list[0]

class CocoDetection(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            # image, target = self.transforms(image, target)
            image = self.transform["eval"](image)

        return image, id# , target

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptionTest(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        init_cap_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        init_captions: list = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)

        self.ids = list(sorted(self.coco.imgs.keys()))[0: 3]

        with open(init_cap_dir) as f:
            if not init_captions:
                init_captions = json.load(f)
            self.captions_dict = {
                caption["image_id"]: caption['caption'] for caption in init_captions
            }


    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_init_caption(self, id) -> List[Any]:
        return self.captions_dict[id]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        init_caption = self._load_init_caption(id)

        if self.transforms:
            # image, target = self.transforms(image, target)
            image = self.transforms(image)

        return image, id, init_caption # , target

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptions(CocoDetection):
    def __init__(self, init_captions):
        super().__init__()
        self.captions_dict = {
            caption["image_id"]: caption['caption'] for caption in init_captions
        }

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]
