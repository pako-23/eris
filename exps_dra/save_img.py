# from torchvision import datasets, transforms
# import PIL.Image as Image
# import os

# def save_cifar10_image(index=88, save_path="cifar10_img88.jpg"):
#     # define simple transform to get PIL images
#     tfm = transforms.ToPILImage()

#     # load CIFAR-10
#     dataset = datasets.CIFAR10(root="./data", train=True, download=True)

#     # get image and label
#     img, label = dataset[index]

#     # save as PNG
#     img.save(save_path, "PNG")
#     print(f"Saved CIFAR-10 image {index} (class={dataset.classes[label]}) to {save_path}")

# # Example usage
# if __name__ == "__main__":
#     save_cifar10_image(88, "cifar10_img88.png")


from torchvision import transforms
import PIL.Image as Image
import os

# If you already have these in your file, you can reuse them
def lfw_dataset(lfw_path, shape_img):
    images_all, labels_all = [], []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    class Dataset_from_Image:
        def __init__(self, imgs, labs, transform=None):
            self.imgs = imgs
            self.labs = labs
            self.transform = transform
    return Dataset_from_Image(images_all, labels_all, transform)

def save_lfw_image(index=0, save_path="lfw_img0.png", lfw_root="./data/lfw"):
    # Build dataset to access original file paths
    ds = lfw_dataset(lfw_root, shape_img=(32, 32))  # shape not used here

    if index < 0 or index >= len(ds.imgs):
        raise IndexError(f"Index {index} out of range (0..{len(ds.imgs)-1}).")

    src_path = ds.imgs[index]  # original JPEG path on disk
    img = Image.open(src_path).convert("RGB")  # ensure 3 channels
    # resize to 32x32
    img = img.resize((32, 32))
    # save as JPEG
    img.save(save_path, "JPEG")
    print(f"Saved LFW image {index} from '{src_path}' to '{save_path}'")

# Example usage
if __name__ == "__main__":
    save_lfw_image(0, "lfw_img0.jpg", lfw_root="./data/lfw")
