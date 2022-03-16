import pandas as pd
from modules.utils import get_countries_geodata, save_file


def main():
    test = pd.read_csv('data/oasis_cross-sectional.csv')
    get_countries_geodata()
    print("main")


if __name__ == '__main__':
    main()
