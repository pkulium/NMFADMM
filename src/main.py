""" Running the NMF-ADMM."""

from admm_nmf import ADMM_NMF
from param_parser import parameter_parser
from utils import read_features, tab_printer

def execute_factorization():
    """
    Reading the target matrix, running optimization and saving to hard drive.
    """
    args = parameter_parser()
    tab_printer(args)
    X = read_features(args.input_path)
    print("\nTraining started.\n")
    model = ADMM_NMF(X, args)
    model.optimize()
    print("\nFactors saved.\n")
    user_factors = model.save_user_factors()
    item_factor = model.save_item_factors()
    return

if __name__ == '__main__':
    execute_factorization()
