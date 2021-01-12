
import pandas as pd
import numpy as np

from unnet.samplers import IdentitySampler

def dummy_tqdm(args):
    return args

class GraphPipeline:
    """A simple Graph pipeline
    
    The generator is used to generate a graphs which are then measures by the measure class
    
    MISSING:
    - uncertainty modification
    - Sampling
    """
    def __init__(self, generator, measure, sampler=IdentitySampler() ):
        self.generator = generator
        self.measure = measure
        self.sampler = sampler
    def execute(self, repetitions=1, progressbar=dummy_tqdm):

        to_iter=range(repetitions)
        if repetitions > 1:
            to_iter = progressbar(range(repetitions))
        final_df = None
        for run_i in to_iter:
            measurements = []
            parameters = []
            for generator_parameters, base_graph in self.generator.execute():#tqdm(self.generator.execute(), total=len(self.generator.parameters)):
                for sampling_parameters, sampled_graph in self.sampler.execute(base_graph):
                    parameters.append({**generator_parameters, ** sampling_parameters})
                    measurements.append(self.measure.execute(sampled_graph))
            df_measurements = pd.DataFrame.from_records(measurements)
            df_parameters = pd.DataFrame.from_records(parameters)
            df = pd.concat([df_parameters, df_measurements], axis=1)
            df['run'] = np.full(len(df), run_i)
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([df, final_df], ignore_index=True)
        return final_df


# Example Classes that wrap around generators samples etc. and iterate through multiple different parameters


class VaryParameters:
    """ Base class for Varying paramters
    
    The two main attributes are 
    1) instance (which is executed multiple times with different parameters)
    2) paramaters (a pandas dataframe where columns indicate parameter names and rows indicate their values)
    """
    def __init__(self, instance, parameters):
        assert hasattr(instance, "execute")
        assert hasattr(instance, "parameters_set")
        self.instance = instance
        assert isinstance(parameters, pd.DataFrame)
        self.parameters = parameters
        

    def execute(self,*args, **kwargs):
        for params in self.parameters.itertuples(index=False):
            self.instance.parameters_set(params._asdict())
            yield (params._asdict(), self.instance.execute(*args, **kwargs))



class ParametersZip(VaryParameters):
    """Vary lists/arrays of parameters in unison
    
    If you only specify a single value for a parameter it is kept constant
    it is required that all lists are of the same length or are just a constant
    """
    def __init__(self, instance, parameters):
        super().__init__(instance, pd.DataFrame.from_dict(parameters))



from itertools import product

class ParametersCombinations(VaryParameters):
    """ Tests all parameter combinations
    
    This class allows you to test all different combinations of the specified parameters
    """
    def __init__(self, instance, parameters):
        self.instance = instance
        parameters_internal=parameters.copy()
        for key, value in parameters.items():
            if isinstance(value, (float, int)):
                parameters_internal[key] = [value]
        
        for key, value in parameters_internal.items():
            assert isinstance(value, (list, tuple, np.array)), f"The key {key} is not iterable"
        
        params = pd.DataFrame.from_records(list(product(*list(parameters_internal.values()))), columns = list(parameters_internal.keys()))
        
        super().__init__(instance, params)
