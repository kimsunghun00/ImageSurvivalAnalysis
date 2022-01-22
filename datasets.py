import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class LoadData:
    def __init__(self, data_folder):
        self.image_path = os.path.join(data_folder, 'all_st_patches_512')
        self.survtime_path = os.path.join(data_folder, 'all_dataset.csv')
        self.grade_path = os.path.join(data_folder, 'grade_data.csv')

    def train_test_split(self, test_size=0.2, random_state=1234):
        labels = self.load_dataset()
        train_labels, test_labels = train_test_split(labels, test_size=test_size, random_state=random_state)
        train_labels, val_labels = train_test_split(train_labels, test_size=test_size, random_state=random_state)

        return train_labels, val_labels, test_labels

    def load_image(self):
        TCGA_ID = []
        image_name = []

        for file_name in os.listdir(self.image_path):
            file_name_split = file_name.split('.')[0]
            tcga_id = file_name_split.split('-')[:3]
            tcga_id = "-".join(tcga_id)

            TCGA_ID.append(tcga_id)
            image_name.append(file_name)
        df = pd.DataFrame({'TCGA ID': TCGA_ID, 'Filename': image_name})
        return df

    def load_dataset(self, interval='months'):
        survtime_df = pd.read_csv(self.survtime_path, usecols=['TCGA ID', 'censored', 'Survival months'])
        if interval == 'day':
            survtime_df['Survival months'] = survtime_df['Survival months'] # day
        elif interval == 'month':
            survtime_df['Survival months'] = survtime_df['Survival months'] / 30.417  # Convert Day to Month
        elif interval == 'year':
            survtime_df['Survival months'] = survtime_df['Survival months'] / 365  # Convert Day to Month

        grade_df = pd.read_csv(self.grade_path, usecols=['TCGA ID', 'Grade', 'Molecular subtype', 'Age at diagnosis'])

        label_df = survtime_df.merge(grade_df, on='TCGA ID')

        return label_df


class PretrainDataset(Dataset):
    """Pathology Image Dataset"""

    def __init__(self, dataframe, root_dir, transform=None):
        """Args :
                dataframe (DataFrame) : pandas DataFrame with annotations.
                root_dir (string) : Directory with all the images.
                transform (callable, optional) : Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        choosen_one = self.df.iloc[idx, :]

        image = Image.open(os.path.join(self.root_dir, choosen_one['Filename']))

        censor = choosen_one['censored']
        censor = torch.tensor(censor, dtype=torch.float)

        survtime = choosen_one['Survival months']
        survtime = torch.tensor(survtime, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'censor': censor, 'survtime': survtime}  # tensor, tensor, tensor

        return sample



class ImageDataset(Dataset):
    """Pathology Image Dataset"""
    
    def __init__(self, dataframe, root_dir, sample_size = 9, transform = None, random_state = None):
        """Args : 
                dataframe (DataFrame) : pandas DataFrame with annotations.
                root_dir (string) : Directory with all the images.
                transform (callable, optional) : Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.root_dir = root_dir
        self.sample_size = sample_size
        self.transform = transform
        self.tcga_id_list = list(self.df['TCGA ID'].unique())
        
        self.dataframes = []        
        for tcga_id in self.tcga_id_list:
            df = self.df.loc[self.df['TCGA ID'] == tcga_id, :].sample(self.sample_size, random_state = random_state)
            self.dataframes.append(df)
            
        
    def __len__(self):
        return len(self.tcga_id_list)

    def get_dataframe(self):
        return self.df
    
    def __getitem__(self, idx):
        choosen_dataframe = self.dataframes[idx]
             
        choosen_images = []
        filenames = list(choosen_dataframe['Filename'].values)
        for filename in filenames:
            image = Image.open(os.path.join(self.root_dir, filename))
            choosen_images.append(image) # PIL image
        
        censor = choosen_dataframe['censored'].values[0]
        censor = torch.tensor(censor, dtype = torch.float)
        
        survtime = choosen_dataframe['Survival months'].values[0]
        survtime = torch.tensor(survtime, dtype = torch.float)
        
        sample = {'image' : choosen_images, 'censor': censor, 'survtime' : survtime} # list of PIL, tensor, tensor
        
        if self.transform:
            transofromed_images = []
            for image in choosen_images:
                image = self.transform(image)
                transofromed_images.append(image)
            transformed_images = torch.stack(transofromed_images)
            
            sample = {'image' : transformed_images, 'censor': censor, 'survtime' : survtime} # tensor, tensor, tensor
        
        return sample


