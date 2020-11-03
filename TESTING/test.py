import argparse, os, json
import commentjson
from utils.utils import ParamParser
from utils.generate_test_data import get_test_data
from utils.test_model import test
from utils.generate_declaration_file import generate_declaration_file
from config import config

parser = argparse.ArgumentParser()
parser.add_argument('--Model_Path', '-p', default = './models/SSL_model_0.pth',help="Full Path to a trained model")
parser.add_argument('--Test_Files','-t', default="./data/testing_files.txt", help="Path to the text file that specifies the testing videos")
parser.add_argument('--Extraction_mode','-m', default="GT" , help="'GT': Ground-truth boxes ; 'YOLO': boxes detected by YOLO")
parser.add_argument('--Non_Target','-NT', default="False" , help="'True': include a Non-Target class; 'False': Do not include a Non-Target class")


def main(model_path, test_list_path, extraction_mode, NT_class):
    test_files = {"path": [], "sampling": []}
    with open(test_list_path, 'r') as f:
        for line in f:
            test_files["path"].append(line.rstrip('\n').split(",")[0])
            test_files["sampling"].append(int(line.rstrip('\n').split(",")[1]))

    # with open(os.path.join("/".join(model_path.split("/")[:-3]), 'params.json')) as json_file:
    #     config = commentjson.load(json_file)

    test_data = get_test_data(extraction_mode, test_files,  NT_class)

    output = test(test_data, model_path, NT_class)

    print("Generating the declaration file ...")
    generate_declaration_file(test_files, output, model_path, args.Extraction_mode, NT_class)

    print("Done")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.Model_Path, args.Test_Files, args.Extraction_mode, args.Non_Target)
    

