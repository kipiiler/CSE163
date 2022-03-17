import pandas as pd
from covid_analysis import run_covid_analysis
from modules.utils import get_countries_geodata, save_file


def main():
    test = pd.read_csv('data/oasis_cross-sectional.csv')
    print("Running Covid analysis")
    run_covid_analysis()
    print("main")


if __name__ == '__main__':
    main()
