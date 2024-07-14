import pandas as pd


def csv_loader(source):
    """
    The csv_loader function imports a csv file from a filepath locally or remotely and returns
    the csv file as a pandas dataframe. The loaded dataframe has the Loan_ID column transformed into the
    index and then, dropped as the final dataframe is imported.
    :param source: path to .csv file.
    :return: dataframe
    """
    # Assign csv File to object df using the pandas method read_csv
    df = pd.read_csv(source)

    # Return the Dataframe
    return df


def excel_loader(source):
    """
    The excel_loader function imports a xlsx file from a filepath locally or remotely and returns
    the Excel file as a pandas dataframe. The loaded dataframe has the Loan_ID column transformed into the
    index and then, dropped as the final dataframe is imported.
    :param source: path to .xlsx file.
    :return: dataframe
    """
    # Assign Excel File to object df using the pandas method read_excel
    df = pd.read_excel(source)

    # Return the Dataframe
    return df