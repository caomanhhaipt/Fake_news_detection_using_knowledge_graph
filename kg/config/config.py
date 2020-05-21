"""
config.py
==========

This module consists of definition of the necessary configuration parameters for all the 
core algorithms. The parameters are seprated into global parameters which are common
across all the algorithms, and local parameters which are specific to the algorithms.

"""

from argparse import ArgumentParser
import importlib

from pykg2vec.utils.kgcontroller import KnowledgeGraph, KGMetaData
from pykg2vec.utils.logger import Logger
from pykg2vec.config.hyperparams import HyperparamterLoader

class Importer:
    """The class defines methods for importing pykg2vec modules.

    Importer is used to defines the maps for the algorithm names and
    provides methods for loading configuration and models.

    Attributes:
        model_path (str): Path where the models are defined.
        config_path (str): Path where the configuration for each models are defineds.
        modelMap (dict): This map transforms the names of model to the actual class names.
        configMap (dict): This map transforms the input config names to the actuall config class names.
    
    Examples:
        >>> from pykg2vec.config.config import Importer
        >>> config_def, model_def = Importer().import_model_config('transe')
        >>> config = config_def()
        >>> model = model_def(config)

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self):
        self.model_path = "pykg2vec.core"
        self.config_path = "pykg2vec.config.config"

        self.modelMap = {"analogy": "ANALOGY.ANALOGY",
                         "complex": "Complex.Complex",
                         "complexn3": "Complex.ComplexN3",
                         "conve": "ConvE.ConvE",
                         "convkb": "ConvKB.ConvKB",
                         "cp": "CP.CP",
                         "hole": "HoLE.HoLE",
                         "distmult": "DistMult.DistMult",
                         "kg2e": "KG2E.KG2E",
                         "kg2e_el": "KG2E.KG2E_EL",
                         "ntn": "NTN.NTN",
                         "proje_pointwise": "ProjE_pointwise.ProjE_pointwise",
                         "rescal": "Rescal.Rescal",
                         "rotate": "RotatE.RotatE",
                         "simple": "SimplE.SimplE",
                         "simple_ignr": "SimplE.SimplE_ignr",
                         "slm": "SLM.SLM",
                         "sme": "SME.SME",
                         "sme_bl": "SME.SME_BL",
                         "transe": "TransE.TransE"}

        self.configMap = {"analogy": "ANALOGYConfig",
                          "complex": "ComplexConfig",
                          "complexn3": "ComplexConfig",
                          "conve": "ConvEConfig",
                          "convkb": "ConvKBConfig",
                          "cp": "CPConfig",
                          "hole": "HoLEConfig",
                          "distmult": "DistMultConfig",
                          "kg2e": "KG2EConfig",
                          "kg2e_el": "KG2EConfig",
                          "ntn": "NTNConfig",
                          "proje_pointwise": "ProjE_pointwiseConfig",
                          "rescal": "RescalConfig",
                          "rotate": "RotatEConfig",
                          "simple": "SimplEConfig",
                          "simple_ignr": "SimplEConfig",
                          "slm": "SLMConfig",
                          "sme": "SMEConfig",
                          "sme_bl": "SMEConfig",
                          "transe": "TransEConfig"}

    def import_model_config(self, name):
      """This function imports models and configuration.

      This function is used to dynamically import the modules within
      pykg2vec. 

      Args:
          name (str): The input to the module is either name of the model or the configuration file. The strings are converted to lowercase to makesure the user inputs can easily be matched to the names of the models and the configuration class.

      Returns:
          object: Configuration and model object after it is successfully loaded.

          `config_obj` (object): Returns the configuration class object of the corresponding algorithm.
          `model_obj` (object): Returns the model class object of the corresponding algorithm.

      Raises:
          ModuleNotFoundError: It raises a module not found error if the configuration or the model cannot be found.
      """
      config_obj = None
      model_obj = None
      try:
          config_obj = getattr(importlib.import_module(self.config_path), self.configMap[name])
          splited_path = self.modelMap[name].split('.')
          model_obj  = getattr(importlib.import_module(self.model_path + ".%s" % splited_path[0]), splited_path[1])

      except ModuleNotFoundError:
          self._logger.error("%s model  has not been implemented. please select from: %s" % (
          name, ' '.join(map(str, self.modelMap.values()))))

      return config_obj, model_obj


class KGEArgParser:
    """The class implements the argument parser for the pykg2vec.

    KGEArgParser defines all the necessary arguements for the global and local 
    configuration of all the modules.

    Attributes:
        general_group (object): It parses the general arguements used by most of the modules.
        general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.
        SME_group (object): It parses the arguments for SME and KG2E algorithms.
        conv_group (object): It parses the arguments for convE algorithms.
        misc_group (object): It prases other necessary arguments.
    
    Examples:
        >>> from pykg2vec.config.config import KGEArgParser
        >>> args = KGEArgParser().get_args()
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        
        ''' arguments regarding TransG '''
        self.TransG_group = self.parser.add_argument_group('TransG function selection')
        self.TransG_group.add_argument('-th', dest='training_threshold', default=3.5, type=float, help="Training Threshold for updateing the clusters.")
        self.TransG_group.add_argument('-nc', dest='ncluster', default=4, type=int, help="Number of clusters")
        self.TransG_group.add_argument('-crp', dest='crp_factor', default=0.01, type=float, help="Chinese Restaurant Process Factor.")
        self.TransG_group.add_argument('-stb', dest='step_before', default=10, type=int, help="Steps before")
        self.TransG_group.add_argument('-wn', dest='weight_norm', default=False, type=lambda x: (str(x).lower() == 'true'), help="normalize the weights!")

        ''' arguments regarding SME and KG2E '''
        self.SME_group = self.parser.add_argument_group('SME KG2E function selection')
        self.SME_group.add_argument('-func', dest='function', default='bilinear', type=str, help="The name of function used in SME model.")
        self.SME_group.add_argument('-cmax', dest='cmax', default=0.05, type=float, help="The parameter for clipping values for KG2E.")
        self.SME_group.add_argument('-cmin', dest='cmin', default=5.00, type=float, help="The parameter for clipping values for KG2E.")

        ''' for conve '''
        # self.conv_group = self.parser.add_argument_group('ConvE specific Hyperparameters')
        # self.conv_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float, help="feature map dropout value used in ConvE.")
        # self.conv_group.add_argument('-idt', dest="input_dropout", default=0.3, type=float, help="input dropout value used in ConvE.")
        # self.conv_group.add_argument('-hdt', dest="hidden_dropout", default=0.3, type=float, help="hidden dropout value used in ConvE.")
        # self.conv_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float, help="The parameter used in label smoothing.")

        '''for convKB'''
        self.convkb_group = self.parser.add_argument_group('ConvKB specific Hyperparameters')
        self.convkb_group.add_argument('-fsize', dest='filter_sizes', default=[1,2,3],nargs='+', type=int, help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.convkb_group.add_argument('-fnum', dest='num_filters', default=50, type=int, help='Filter numbers to be used in convKB')

        '''for RotatE'''
        self.rotate_group = self.parser.add_argument_group('RotatE specific Hyperparameters')
        self.rotate_group.add_argument('-al', dest='alpha', default=0.1, type=float, help='The alpha used in self-adversarial negative sampling.')

        ''' arguments regarding hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda for regularization.')
        self.general_hyper_group.add_argument('-b',   dest='batch_training', default=128, type=int, help='training batch size')
        self.general_hyper_group.add_argument('-mg',  dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str, help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s',   dest='sampling', default='uniform', type=str, help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-ngr', dest='negrate', default=1, type=int, help='The number of negative samples generated per positve one.')
        self.general_hyper_group.add_argument('-l',   dest='epochs', default=100, type=int, help='The total number of Epochs')
        self.general_hyper_group.add_argument('-lr',  dest='learning_rate', default=0.01, type=float,help='learning rate')
        self.general_hyper_group.add_argument('-k',   dest='hidden_size', default=50, type=int,help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km',  dest='ent_hidden_size', default=50, type=int, help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr',  dest='rel_hidden_size', default=50, type=int, help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-k2',  dest='hidden_size_1', default=10, type=int, help="Hidden embedding size for relations.")

        self.general_hyper_group.add_argument('-l1',  dest='l1_flag', default=True, type=lambda x: (str(x).lower() == 'true'),help='The flag of using L1 or L2 norm.')

        ''' working environments '''
        self.environment_group = self.parser.add_argument_group('Working Environments')
        self.environment_group.add_argument('-gp',  dest='gpu_frac', default=0.8, type=float, help='GPU fraction to use')
        self.environment_group.add_argument('-npg', dest='num_process_gen', default=2, type=int, help='number of processes used in the Generator.')

        ''' basic configs '''
        self.general_group = self.parser.add_argument_group('Generic')
        self.general_group.add_argument('-mn',    dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db',    dest='debug',      default=False, type=lambda x: (str(x).lower() == 'true'), help='To use debug mode or not.')
        self.general_group.add_argument('-exp',   dest='exp', default=False, type=lambda x: (str(x).lower() == 'true'), help='Use Experimental setting extracted from original paper. (use with -ds or FB15k in default)')
        self.general_group.add_argument('-ds',    dest='dataset_name', default='Freebase15k', type=str, help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
        self.general_group.add_argument('-dsp',   dest='dataset_path', default="../dataset/CRAWL", type=str, help='The path to custom dataset.')
        self.general_group.add_argument('-ld',    dest='load_from_data', default=False, type=lambda x: (str(x).lower() == 'true'), help='load from tensroflow saved data!')
        self.general_group.add_argument('-sv',    dest='save_model', default=True, type=lambda x: (str(x).lower() == 'true'), help='Save the model!')
        self.general_group.add_argument('-tn',    dest='test_num', default=1000, type=int, help='The total number of test triples')
        self.general_group.add_argument('-ts',    dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_group.add_argument('-t',     dest='tmp', default='../intermediate', type=str,help='The folder name to store trained parameters.')
        self.general_group.add_argument('-r',     dest='result', default='../results', type=str,help="The folder name to save the results.")
        self.general_group.add_argument('-fig',   dest='figures', default='../figures', type=str,help="The folder name to save the figures.")
        self.general_group.add_argument('-plote', dest='plot_embedding', default=False,type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-plot',  dest='plot_entity_only', default=False,type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')

    def get_args(self, args):
      """This function parses the necessary arguments.

      This function is called to parse all the necessary arguments. 

      Returns:
          object: ArgumentParser object.
      """
      return self.parser.parse_args(args)


class BasicConfig:
    """The class defines the basic configuration for the pykg2vec.

    BasicConfig consists of the necessary parameter description used by all the 
    modules including the algorithms and utility functions.

    Args:
      test_step (int): Testing is carried out every test_step.
      test_num (int): Number of triples that will be tested during evaluation.
      triple_num (int): Number of triples that will be used for plotting the embedding.
      tmp (Path Object): Path where temporary model information is stored.
      result (Path Object): Gives the path where the result will be saved.
      figures (Path Object): Gives the path where the figures will be saved.
      gpu_fraction (float): Amount of GPU fraction that will be made available for training and inference.
      gpu_allow_growth (bool): If True, allocates only necessary GPU memory and grows as required later.
      loadFromData (bool): If True, loads the model parameters if available from memory.
      save_model (True): If True, store the trained model parameters.
      disp_summary (bool): If True, display the summary before and after training the algorithm.
      disp_result (bool): If True, displays result while training.
      plot_embedding (bool): If True, will plot the embedding after performing t-SNE based dimensionality reduction.
      log_training_placement (bool): If True, allows us to find out which devices the operations and tensors are assigned to.
      plot_training_result (bool): If True, plots the loss values stored during training.
      plot_testing_result (bool): If True, it will plot all the testing result such as mean rank, hit ratio, etc.
      plot_entity_only (bool): If True, plots the t-SNE reduced embdding of the entities in a figure.
      full_test_flag (bool): It True, performs a full test after completing the training for full epochs.
      hits (List): Gives the list of integer for calculating hits.
      knowledge_graph (Object): It prepares and holds the instance of the knowledge graph dataset.
      kg_meta (object): Stores the statistics metadata of the knowledge graph.
    
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, args):

        # Training and evaluating related variables
        self.test_step = args.test_step
        self.full_test_flag = (self.test_step == 0)
        self.test_num = args.test_num
        self.hits = [1, 3, 5, 10]
        self.loadFromData = args.load_from_data
        self.save_model = args.save_model
        self.disp_summary = True
        self.disp_result = False
        
        self.patience = 3 # should make this configurable as well.
        
        # Visualization related, 
        # p.s. the visualizer is disable for most of the KGE methods for now. 
        self.disp_triple_num = 20
        self.plot_embedding = args.plot_embedding
        self.plot_training_result = True
        self.plot_testing_result = True
        self.plot_entity_only = args.plot_entity_only
        
        # Working environment variables.
        self.num_process_gen = args.num_process_gen
        self.log_device_placement = False
        self.gpu_fraction = args.gpu_frac
        self.gpu_allow_growth = True
        # self.gpu_config = tf.ConfigProto(log_device_placement=self.log_device_placement)
        # self.gpu_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
        # self.gpu_config.gpu_options.allow_growth = self.gpu_allow_growth

        # Knowledge Graph Information
        self.custom_dataset_path = args.dataset_path
        self.knowledge_graph = KnowledgeGraph(dataset=self.data, custom_dataset_path=self.custom_dataset_path)
        self.kg_meta = self.knowledge_graph.kg_meta
        
        # The results of training will be stored in the following folders 
        # which are relative to the parent folder (the path of the dataset).
        dataset_path = self.knowledge_graph.dataset.dataset_path
        self.path_tmp =  dataset_path / 'intermediate'
        self.path_tmp.mkdir(parents=True, exist_ok=True)
        self.path_result = dataset_path / 'results'
        self.path_result.mkdir(parents=True, exist_ok=True)
        self.path_figures = dataset_path / 'figures'
        self.path_figures.mkdir(parents=True, exist_ok=True)
        self.path_embeddings = dataset_path / 'embeddings'
        self.path_embeddings.mkdir(parents=True, exist_ok=True)

        # debugging information 
        self.debug = args.debug

    def summary(self):
        """Function to print the summary."""
        summary = []
        summary.append("")
        summary.append("------------------Global Setting--------------------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.__dict__.keys()])) +20
        for key, val in self.__dict__.items():
            if key in self.__dict__['hyperparameters']:
                continue

            if isinstance(val, (KGMetaData, KnowledgeGraph)) or key.startswith('gpu') or key.startswith('hyperparameters'):
                continue

            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            summary.append("%s : %s"%(key, val))
        summary.append("---------------------------------------------------")
        summary.append("")
        self._logger.info("\n".join(summary))

    def summary_hyperparameter(self, model_name):
        """Function to print the hyperparameter summary."""
        summary_hyperparameter = []
        summary_hyperparameter.append("")
        summary_hyperparameter.append("-----------%s Hyperparameter Setting-------------"%(model_name))
        maxspace = len(max([k for k in self.hyperparameters.keys()])) + 15
        for key,val in self.hyperparameters.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            summary_hyperparameter.append("%s : %s" % (key, val))
        summary_hyperparameter.append("---------------------------------------------------")
        self._logger.info("\n".join(summary_hyperparameter))


class TransEConfig(BasicConfig):
    """This class defines the configuration for the TransE Algorithm.

    TransEConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for both entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transe')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)



