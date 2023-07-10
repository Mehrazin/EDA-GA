import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
from collections import Counter
import re
warnings.filterwarnings('ignore')

class DataOverview:
    """
        Provides an overview of a given pandas DataFrame,
        including the percentage of nulls for each column,
        the percentage of unique values for each column, etc.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def null_count(self):
        return self.data.isna().sum().values

    def percent(self, array):
        return [f"{round((value / self.data.shape[0]) * 100, 2)}%" for value in array]

    def overview(self):
        cols = self.data.columns.tolist()
        count = self.data.count().values
        unique = [self.data[col].nunique() for col in cols]
        null_values = self.null_count()

        new_data = pd.DataFrame(
            data=list(zip(
                cols,
                count,
                null_values,
                self.percent(null_values),
                unique,
                self.percent(unique),
                self.data.dtypes)),
            columns=[
                'Column',
                'Count',
                'Missing_value_count',
                'Missing_value_percentage',
                'Unique_value_count',
                'Unique_value_percentage',
                'Dtype']
        )
        return new_data

def mixed_type_columns(data):
    mixed_cols = {'Column': [], 'Data type': []}
    for column in data.columns:
        type = pd.api.types.infer_dtype(data[column])
        if type.startswith("mixed"):
            mixed_cols['Column'].append(column)
            mixed_cols['Data type'].append(type)
    if not mixed_cols['Column']:
        print('No columns contain mixed types.')
    else:
        print(pd.DataFrame(mixed_cols))

def missing_values_percent(data):
    total_cells = np.product(data.shape)
    total_missing = data.isnull().sum().sum()
    print("The dataset contains", round(((total_missing/total_cells) * 100), 2), "%", "missing values.")

def missing_values_info(data):
    mis_vals = data.isnull().sum()
    mis_vals_percent = 100 * mis_vals / len(data)
    mis_vals_table = pd.concat([mis_vals, mis_vals_percent, data.dtypes], axis=1).rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    mis_vals_table = mis_vals_table[mis_vals_table.iloc[:,0] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)

    print ("Your dataframe has " + str(data.shape[1]) + " columns.\n"
            "There are " + str(mis_vals_table.shape[0]) +
            " columns that have missing values.")

    if mis_vals_table.shape[0] == 0:
        return

    return mis_vals_table

def convert_to_str(data, cols):
    for column in cols:
        data[column] = data[column].astype("string")

def remove_duplicates(data):
    old_size = data.shape[0]
    data.drop_duplicates(inplace=True)
    new_size = data.shape[0]
    diff = old_size - new_size
    print(f"{'No duplicate' if diff == 0 else f'{diff}'} rows {'found' if diff == 0 else 'removed'}.")

def plot_distribution(data:pd.DataFrame, col:str, color:str='cornflowerblue'):
    sns.displot(data=data, x=col, color=color,  kde=False, height=6, aspect=2)
    plt.title(f'Distribution of {col}', size=20, fontweight='bold')
    plt.show()

def plotly_bar(data, x_col, y_col, title=None, image_width=800, color=['cornflowerblue']):
    title = title or f'Distribution of {x_col}'
    fig = px.bar(data, x=x_col, y=y_col, title=title)
    return Image(pio.to_image(fig, format='png', width=image_width))

def plotly_pie(data, col, title=None, image_width=800, min_count=None):
    df = data.groupby([col]).size().reset_index().rename(columns={0: 'count'}).sort_values("count", ascending=False)
    if min_count:
        df.loc[df['count'] < min_count, col] = f'Other {col}s'
    title = title or f'Distribution of {col}'
    fig = px.pie(df, values='count', names=col, title=title)
    return Image(pio.to_image(fig, format='png', width=image_width))


# Function to check and calculate percentage
def check_percentage(df, regex_pattern):
    regex = re.compile(regex_pattern)
    match = df['name'].apply(lambda x: bool(regex.search(x)))
    percentage = match.sum() / len(df) * 100
    return round(percentage, 2)

def extract_city(locality):
    if pd.isnull(locality):
        return '<NA>'
    # extract cities from locality
    geo = geotext.GeoText(locality).cities
    if len(geo) > 0:
        return geo[0]
    return locality.split(',')[0]

def categorize(size):
    switcher={
        '1 - 10': 1,
        '11 - 50': 2,
        '51 - 200': 3,
        '201 - 500': 4,
        '501 - 1000': 5,
        '1001 - 5000': 6,
        '5001 - 10000': 7,
        '10001+': 8
    }
    return switcher.get(size, 0)


def industry_in_name(row):
    puncs = [punc for punc in string.punctuation]
    puncs.append('and')
    industry_words = set(filter(lambda x: x not in puncs, re.split(r'\s|-', row['industry'])))
    name_words = re.split(r'\s|-', row['name'])
    return len([i for i in name_words if i in industry_words])


def locality_in_name(row):
    if pd.isnull(row['locality']):
        return np.nan
    puncs = [punc for punc in string.punctuation]
    puncs.append('and')
    locality_words = set(filter(lambda x: x not in puncs, re.split(r'\s|-', row['locality'])))
    name_words = re.split(r'\s|-', row['name'])
    return len([i for i in name_words if i in locality_words])

def year_in_name(row):
    if (row['year founded'] == 0):
        return None
    year = row['year founded']
    regexp = re.compile(f'{year}')
    return regexp.search(row['name'])
