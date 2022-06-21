import argparse

def get_parser():
    # yapf: disable
    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer.")

    ########################################
    # Added related parameters about MG
    parser.add_argument('--setloss', type=str, required=False, default="ce+distill", 
                        help='set loss: ce/ce+distill/ce+hierloss/ce+distill+hierloss/distill+hierloss')
    parser.add_argument('--vis_hier', action='store_true', 
                        help='use visual feature to build hierarchy instead of semantic information')
    parser.add_argument('--cluster', type=int, required=False, default=-2, 
                        help='If you use this parameter, the network will use word embedding to generate the hierarchical information')
    parser.add_argument('--mg_beta', type=float, required=False, default=50.0,
                        help='param for hier loss')
    parser.add_argument('--mg_scale', type=float, required=False, default=0.01,
                        help='scale for hier loss')
    # End
    ########################################