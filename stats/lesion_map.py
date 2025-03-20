import ants
import numpy as np
import multiprocessing

import datasets.dataset_loaders as dataset_loaders

def generate_stat_map(dataset: dataset_loaders.Subject, template: ants.ants_image.ANTsImage, output_file: str) -> None:
    """
    Computes a statistical map of the lesion probability given a dataset of subjects.
    
    Parameters:
        dataset (list[dataset_loaders.Subject]): The list of subjects to be used for the statistical map.
        template (ants.ants_image.ANTsImage): The template image for registration to MNI space.
        output_file (str): The path where the computed statistical map will be saved.
    
    Returns:
        None
    """
    stat_map = np.zeros(template.shape)

    for i, subj in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {subj.name}...")
        subj.load_data()

        subj.extract_brain()
        subj.apply_transform_to_mni()
        subj.space_integrity_check()
        subj.empty_label_check()

        stat_map += subj.label.numpy()

        subj.free_data()
    ants.image_write(ants.new_image_like(template, stat_map), output_file)

if __name__ == "__main__":
    template = ants.image_read("datasets/template_flair_mni.nii.gz")

    dataset = dataset_loaders.ISLES2022()
    p = multiprocessing.Process(target=generate_stat_map, args=[dataset, template, "results/stat_map_ISLES22.nii.gz"])
    p.start()