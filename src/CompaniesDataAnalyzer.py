import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract
from text_unidecode import unidecode
from cleanco import basename


class CompaniesDataAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: pd.DataFrame = None
        self.key_fields = []


    def load_data(self):
        """Load the parquet file into a DataFrame"""
        self.df = pd.read_parquet(self.file_path)
        return self
    

    def plot_values(self, min_present_ratio=0.5):
        """
        Plots the percentage of non-null (present) values for each column in the dataset.

        Parameters:
        min_present_ratio (float): Minimum percentage (0 to 1) of non-null values required for a column to be included in the plot.
        """

        if self.df is None or self.df.empty:
            self.load_data()

        # The percentage of non-null values per column
        non_null_ratio = (1 - self.df.isnull().mean())

        # Filter columns based on the threshold
        filtered_columns = non_null_ratio[non_null_ratio >= min_present_ratio].sort_values(ascending=False)

        # Plot the results
        plt.figure(figsize=(20, 8))
        sns.barplot(x=filtered_columns.values, y=filtered_columns.index, palette="viridis", orient='h', hue=filtered_columns.index, legend=False)
        plt.xlabel('Percentage of Non-Null Values')
        plt.ylabel('Columns')
        plt.title('Percentage of Non-Null Values per Column')
        plt.show()

    def set_key_fields(self, key_fields):
        """Set the key fields that should be prioritized in analysis."""
        self.key_fields = key_fields


    def handle_missing_values(self):
        """
        Handle missing values in the DataFrame by extracting data from other fields.
        """
        if self.df is None:
            self.load_data()
        
        # Create a mask for rows where website_domain is null and website_url is not null
        mask = self.df["website_domain"].isnull() & self.df["website_url"].notnull()
        if mask.any():
            self.df.loc[mask, "website_domain"] = self.df.loc[mask, "website_url"].apply(
                lambda x: tldextract.extract(x).domain if pd.notnull(x) else None
            )

        mask = self.df["phone_numbers"].isnull() & self.df["primary_phone"].notnull()
        if mask.any():
            self.df.loc[mask, "phone_numbers"] = self.df.loc[mask, "primary_phone"]
        
        mask = self.df["main_address_raw_text"].isnull()
        if mask.any():
            components = []
            for field in ["main_street_number", "main_street", "main_city", "main_region", "main_country"]:
                if field in self.df.columns:
                    components.append(self.df[field].fillna(""))
        
            combined_address = components[0]
            for comp in components[1:]:
                combined_address = combined_address.str.cat(comp, sep=" ", na_rep="")
            
            self.df.loc[mask, "main_address_raw_text"] = combined_address.loc[mask].str.strip()


    def _clean_text_field(self, series: pd.Series, isCompany: bool) -> pd.Series:
        """
        Cleans a text field in the DataFrame.
        """
        if isCompany:
            return (
                series
                .fillna("")
                .apply(lambda x: unidecode(x)) # Convert non-ASCII characters to ASCII
                .apply(lambda x: basename(x)) # Remove legal suffixes
                .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True) # Remove special characters
                .str.lower()
            )
        else:
            return (
                series
                .fillna("")
                .str.lower()
                .apply(lambda x: unidecode(x)) # Convert non-ASCII characters to ASCII
                .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True) # Remove special characters
                .str.replace(r"\s+", " ", regex=True) # Remove multiple extra spaces
                .str.strip()
            )
        

    def clean_data(self):
        """
        Cleans the DataFrame by applying text cleaning to specific string fields.
        """
        for field in ["company_name", "main_street", "main_city", "main_region", "main_country", "main_industry"]:
            if field in self.df.columns:
                self.df[field] = self._clean_text_field(self.df[field], "company" in field)


    def print_key_fields(self, num=5):
        key_fields_df = pd.DataFrame(self.df[self.key_fields].head(num))
        return key_fields_df
