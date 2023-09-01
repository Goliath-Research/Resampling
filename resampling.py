# This version uses numpy arrays instead of pandas DataFrames. It is faster!

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
#from itertools import combinations
from abc import ABC, abstractmethod

#import seaborn as sns



class Samples(ABC):
    '''
    Abstract class for generating samples using a resampling method 
    '''

    def __init__(self, sample_data: np.ndarray, num_samples: int = 10000):
        '''    
        It creates the sample distribution using a resampling (bootstrap or permutation)
        sample_data:    sample data (must be representative of the population)        
        num_samples:    number of samples to generate for the bootstrap method        
        '''
        self._sample_data: np.ndarray = sample_data     # sample data   
        self._num_samples: int = num_samples            # number of samples to generate
        self._sample_size: int = len(self._sample_data) # sample size        
        self._samples: np.ndarray = np.zeros((self._sample_size, self._num_samples))
        
        self.generate_samples()

    @abstractmethod
    def generate_samples(self):
        pass

    @property
    def sample_data(self) -> np.ndarray:
        '''Sample of the original population (np.ndarray)'''
        return self._sample_data

    @property
    def sample_size(self) -> int:
        '''Sample size (positive integer)'''
        return self._sample_size

    @property
    def num_samples(self) -> int:
        '''Number of samples (positive integer)'''
        return self._num_samples

    @property
    def samples(self) -> pd.DataFrame:
        '''Getter method for accessing the DataFrame of _samples'''
        return self._samples  
        



class BSamples(Samples):
    '''
    Class for generating samples using the bootstrap method (resampling WITH replacement)    
    '''    

    def generate_samples(self):
        '''
        It returns a DataFrame where each column (num_samples columns) is a sample with replacement.
        It uses _sample_data to generate the samples.
        '''                  
        # Generating the samples WITH replacement
        sample_boot = np.random.choice(self._sample_data, replace=True,
                                       size=(self._sample_size, self._num_samples))
        # Now, sample_boot is a 2D NumPy array with shape (self._sample_size, self._num_samples)
        self._samples = sample_boot
        
                         

    

class PSamples(Samples):
    '''
    Class for generating samples using the permutation method (resampling WITHOUT replacement)    
    '''
    
    def generate_samples(self):
        '''
        It returns a 2D NumPy array where each column is a sample without replacement.        
        '''
        # Create a matrix by repeating the _sample_data num_samples times
        replicated_data = np.tile(self._sample_data, (self._num_samples, 1)).T        
        # Apply permutation on each column of the matrix
        self._samples = np.apply_along_axis(np.random.permutation, 0, replicated_data) 
    

    






class HypothesisTest(ABC):
    '''
    Abstract class for hypothesis testing using resampling methods (bootstrap and permutation) 
    '''

    def __init__(self):
        '''
        It initializes the class. 
        HypothesisTest class computes the p-value given a sample distribution and an observed statistic.        
        '''
        self._sample_distribution: np.ndarray = self.get_sample_distribution()
        self._obs_stat: float = self.get_observed_stat()
        self._p_value:  float = 1.0

    @abstractmethod
    def get_sample_distribution(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_observed_stat(self) -> float:
        pass

    def get_p_value(self, alternative='two-sided') -> float:
        '''
        It returns the p-value.                
        alternative: 'two-sided', 'smaller', or 'larger'    
        '''         
        ecdf = ECDF(self._sample_distribution)         
        if alternative=='two-sided':
            p_val = 2 * min(ecdf(self._obs_stat), 1 - ecdf(self._obs_stat))            
        elif alternative=='smaller':
            p_val = ecdf(self._obs_stat)
        else:   #alternative=='larger'
            p_val = 1-ecdf(self._obs_stat)
        #print('stat = %.2f    p-value: %.4f' %(self._obs_stat, p_val))
        return p_val
    



class BootstrapIndependentTest(HypothesisTest):
    '''
    Class for independent samples hypothesis testing using the bootstrap method.    
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class                
        '''   
        # Verifying the inputs
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 arrays are required.")   
        
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError("Failed to convert inputs to NumPy arrays.")   
        
        # Converting arrays to Sample objects
        self._sample_objects = [BSamples(array) for array in arrays]

        # Getting the overall size
        self._overall_size: int = sum([s.sample_size for s in self._sample_objects])

        # Getting the overall mean                        
        self._overall_mean: float = np.concatenate(self._arrays).mean()

        # Initializing the parent class
        super().__init__()               


    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        Difference in means statistic
        '''               
        # Getting the two sample objects
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        # Shifting the samples for sharing the mean
        arr0 = s0.samples - s0.sample_data.mean() + self._overall_mean 
        arr1 = s1.samples - s1.sample_data.mean() + self._overall_mean    
        # Returning the difference in means   
        return arr0.mean(axis=0) - arr1.mean(axis=0)
            
        
    def get_sample_distribution_k(self) -> np.ndarray:    
        '''
        The sampling distribution will be the square of... k samples (k>2)
        '''     
        # Shifting the samples for sharing the mean
        arr_shifted = [self._sample_objects[k].samples - self._sample_objects[k].sample_data.mean() 
                            + self._overall_mean for k in range(self._k)]
        # Computing the statistic
        sample_dist = sum([((arr_shifted[k].mean(axis=0) - self._overall_mean)**2) *  
                            (self._sample_objects[k].sample_size / self._overall_size) 
                            for k in range(self._k)])                        
        return np.sqrt(sample_dist)

    
    def get_sample_distribution(self) -> np.ndarray:
        '''
        It returns a np.array with the sample distribution based on means.
        '''          
        arr = self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
        #sns.histplot(arr, stat='density', color = 'darkblue', edgecolor='black', linewidth=1)
        return arr
        
        
    def get_observed_stat_2(self) -> float:
        '''
        Difference in means statistic
        '''                
        s0 = self._sample_objects[0]
        s1 = self._sample_objects[1]        
        return s0.sample_data.mean() - s1.sample_data.mean()        
        
    
    def get_observed_stat_k(self) -> float:
        '''
        The observed statistic will be will be the square of... k samples (k>2)
        '''                  
        stat = sum([(self._sample_objects[k].sample_data.mean() - self._overall_mean)**2 *
                            (self._sample_objects[k].sample_size / self._overall_size) 
                            for k in range(self._k)])        
        return np.sqrt(stat) 
    

    def get_observed_stat(self) -> float:
        '''
        It returns the observed statistic.        
        '''
        stat = self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
        return stat




class BootstrapRelatedTest(HypothesisTest):
    '''
    Class for related samples hypothesis testing using the bootstrap method.
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class  
        '''   
        # Verifying the inputs            
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")     
        
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError('Failed to convert inputs to NumPy arrays.')    
        
        # All arrays must have the same size
        self._sample_size = len(arrays[0])
        try:
            assert all(len(sample) == self._sample_size for sample in self._arrays)
        except:            
            raise ValueError("All samples must be the same length.")    
        # The block resampling procedure respects the observation pairing of 
        # blocks of related observations.   
              
        # Generating indexes        
        self._boot_indexes = BSamples(np.arange(self._sample_size)) 
        
        # Getting the overall mean
        self._overall_mean: float = np.concatenate(self._arrays).mean()
        
        # Initializing the parent class
        super().__init__()          
             
        
                 

    def mean_difference(self, data: np.ndarray) -> float: 
        '''
        It computes the mean of the absolute value of the differences of all pairs of samples.
        '''           
        means = data.mean(axis=0)                      
        return np.mean(np.abs(np.subtract.outer(means, means)[np.triu_indices(len(means), k=1)]))                


    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        Difference in means statistic
        '''      
        # Getting the two samples
        s0 = self._arrays[0][self._boot_indexes.samples]
        s1 = self._arrays[1][self._boot_indexes.samples]        
        # Shifting the samples for sharing the mean
        s0_shifted = s0 - self._arrays[0].mean() + self._overall_mean 
        s1_shifted = s1 - self._arrays[1].mean() + self._overall_mean              
        return s0_shifted.mean(axis=0) - s1_shifted.mean(axis=0)
        
        


    def get_sample_distribution_k(self) -> np.ndarray:
        '''
        It gets the sample distribution based on mean_difference method.  
        '''     
        # Getting the k samples
        s = [self._arrays[i][self._boot_indexes.samples] for i in range(self._k)]
        # Shifting the samples for sharing the mean
        s_shifted = [s[i] - self._arrays[i].mean() + self._overall_mean for i in range(self._k)]
        stacked_arrays = np.stack(s_shifted, axis=0)         
        transposed_arrays = np.transpose(stacked_arrays, (2, 0, 1))
        sample_dist = np.array([self.mean_difference(transposed_arrays[i]) 
                       for i in range(transposed_arrays.shape[0])])
        return sample_dist      

        
    
    


    def get_sample_distribution(self) -> np.ndarray:
        '''
        It returns a np.array with the sample distribution based on means.
        '''          
        arr = self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
        #sns.histplot(arr, stat='density', color = 'skyblue', edgecolor='white', linewidth=1)
        return arr
      
        
    def get_observed_stat(self) -> float:
        '''
        The observed statistic is mean_difference method computed on the actual observed data. 
        '''     
        data = np.array(self._arrays).T    
        return self.mean_difference(data)










class PermutationIndependentTest(HypothesisTest):
    '''
    Class for independent samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class                
        '''   
        # Verifying the inputs        
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")   
        
        # Converting inputs to numpy arrays
        try:
            self._arrays = [np.array(array) for array in arrays]
        except:
            raise ValueError("Failed to convert inputs to NumPy arrays.")  
        
        # Getting every sample size
        self._sample_sizes: np.array = np.array([len(s) for s in self._arrays])        
        
        # Getting the shuffled np.ndarray
        self._sample_objects = PSamples(np.concatenate(self._arrays))

        # Getting a list of _k shuffled NumPy arrays 
        self._arrays_list = np.split(self._sample_objects.samples, np.cumsum(self._sample_sizes)[:-1])        

        # Getting the overall size
        self._overall_size: int = self._sample_sizes.sum()        

        # Getting the overall mean                        
        self._overall_mean: float = np.concatenate(self._arrays).mean()        

        # Initializing the parent class
        super().__init__()     



    def get_sample_distribution_2(self) -> np.ndarray:
        '''
        Difference in means statistic
        '''   
        # Getting the two sample objects
        arr0 = self._arrays_list[0]
        arr1 = self._arrays_list[1]   
        # Computing the difference in means
        sample_dist = arr0.mean(axis=0) - arr1.mean(axis=0)        
        return sample_dist
            
        
    def get_sample_distribution_k(self) -> np.ndarray:    
        '''
        The sampling distribution will be the square of... k samples (k>2)
        '''        
        sample_dist = sum([((self._arrays_list[k].mean(axis=0) - self._overall_mean)**2) *  
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                  
        return np.sqrt(sample_dist)


    def get_sample_distribution(self) -> np.ndarray:
        '''
        It returns a np.array with the sample distribution based on means.
        '''      
        arr = self.get_sample_distribution_2() if self._k == 2 else self.get_sample_distribution_k()
        #sns.histplot(arr, stat='density', color = 'firebrick', edgecolor='grey', linewidth=1)
        
        return arr
        

        
    def get_observed_stat_2(self) -> float:
        '''
        Difference in means statistic
        '''     
        return self._arrays[0].mean() - self._arrays[1].mean()                         
        
        
    
    def get_observed_stat_k(self) -> float:
        '''
        The observed statistic will be will be the square of... k samples (k>2)
        '''                  
        stat = sum([(self._arrays[k].mean() - self._overall_mean)**2 * 
                            (self._sample_sizes[k] / self._overall_size) 
                            for k in range(self._k)])                
        return np.sqrt(stat) 
    

    def get_observed_stat(self) -> float:
        '''
        It returns the observed statistic.        
        '''
        stat = self.get_observed_stat_2() if self._k == 2 else self.get_observed_stat_k()        
        return stat
    








class PermutationRelatedTest(HypothesisTest):
    '''
    Class for related samples hypothesis testing using the permutation method.    
    '''

    def __init__(self, *arrays):
        '''
        Verifying the inputs and initializing the class       
        '''   
        # Verifying the inputs
        self._k = len(arrays)
        if self._k < 2:
            raise ValueError("At least 2 samples are required.")  
            
        # Converting inputs to numpy arrays 
        try:
            self._arrays = [np.array(array) for array in arrays]            
        except:            
            raise ValueError("Failed to convert inputs to NumPy arrays.")
        
        # All arrays must have the same size
        self._sample_size = len(arrays[0])
        try:
            assert all(len(sample) == self._sample_size for sample in self._arrays)
        except:            
            raise ValueError("All samples must be the same length.")    
                
        # The block resampling procedure respects the observation pairing of 
        # blocks of related observations.        
        self._sample_objects = [PSamples(pair) for pair in zip(*self._arrays)]     
                
        # Initializing the parent class
        super().__init__()   



    def mean_difference(self, data: np.ndarray) -> float: 
        '''
        It computes the mean of the absolute value of the differences of all pairs of samples.
        '''           
        means = data.mean(axis=0)                  
        return np.mean(np.abs(np.subtract.outer(means, means)[np.triu_indices(len(means), k=1)]))                


    def get_sample_distribution(self) -> np.ndarray:
        '''
        It gets the sample distribution for k samples (k>2) based on mean_difference method.  
        '''    
        arrays_list = [self._sample_objects[i].samples for i in range(5)]
        stacked_arrays = np.stack(arrays_list, axis=0)
        transposed_arrays = np.transpose(stacked_arrays, (2, 0, 1))
        sample_dist = np.array([self.mean_difference(transposed_arrays[i]) 
                       for i in range(transposed_arrays.shape[0])])
        #sns.histplot(sample_dist, stat='density', color = 'lightcoral', edgecolor='grey', linewidth=1)        
        return sample_dist      
      
        
    def get_observed_stat(self) -> float:
        '''
        The observed statistic is mean_difference method computed on the actual observed data. 
        '''     
        data = np.array(self._arrays).T    
        return self.mean_difference(data)
    
    



       