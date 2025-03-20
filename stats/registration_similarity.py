import ants
import pandas
import datasets.dataset_loaders as dataset_loaders
import multiprocessing

def compute_similarity(image1, image2, queue):
    """
    Compute similarity between two ANTs images and put the result in a multiprocessing queue.

    Parameters
    ----------
    image1 : ants.ANTsImage
        First image
    image2 : ants.ANTsImage
        Second image
    queue : multiprocessing.Queue
        Queue to put the result in
    """
    similarity = ants.image_similarity(image1, image2)
    queue.put(similarity)

def registration_measure(dataset: list[dataset_loaders.Subject], dataset_name: str, results_df: pandas.DataFrame):
        """
        Compute similarity metrics between DWI and FLAIR (and original DWI) after registration.
        
        Parameters:
            dataset (list[dataset_loaders.Subject]): List of subjects with MRI data.
            dataset_name (str): Name of the dataset.
            results_df (pandas.DataFrame): DataFrame where the results will be stored.
        """
        queue = multiprocessing.Queue()
        for i, subj in enumerate(dataset):
            print(f"{dataset_name} {i+1}/{len(dataset)}: {subj.name}")
            
            subj.load_data(load_label=False, transform_to_flair=False)
            dwi_orig = subj.dwi
            subj.free_data()
            
            subj.load_data(load_label=False)
            dwi_tf = subj.dwi

            # DWI to original DWI
            mutual_information = ants.metrics.image_mutual_information(dwi_tf, dwi_orig)

            # compute similarity in another thread
            # because runnning compute similarity multiple times causes growing memory usage
            # there is probably memory leak in the ANTs
            p = multiprocessing.Process(target=compute_similarity, args=(dwi_tf, dwi_orig, queue))
            p.start()
            p.join()
            similarity = queue.get()
            results_df.loc[len(results_df)] = [dataset_name, subj.name, 'DWI-DWI', mutual_information, similarity]

            # dwi to flair
            mutual_information = ants.metrics.image_mutual_information(dwi_tf, subj.flair)
            p = multiprocessing.Process(target=compute_similarity, args=(dwi_tf, subj.flair, queue))
            p.start()
            p.join()
            similarity = queue.get()
            results_df.loc[len(results_df)] = [dataset_name, subj.name, 'DWI-FLAIR', mutual_information, similarity]

            # invariant_similarity = ants.invariant_image_similarity(dwi_tf, dwi_orig)
            # module 'ants' has no attribute 'invariant_image_similarity'

            subj.free_data()

if __name__ == "__main__":
    results_df = pandas.DataFrame(columns=['Dataset', 'Subject', 'Type', 'Mutual Information', 'Similarity'])

    registration_measure(dataset_loaders.ISLES2022(), "ISLES2022", results_df)

    # save results
    results_df.to_csv("results/registration_similarity_prueba.csv", index=False) 