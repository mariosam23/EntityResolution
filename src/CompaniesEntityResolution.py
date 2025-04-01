import pandas as pd
from itertools import combinations
from jellyfish import jaro_winkler_similarity
from Levenshtein import ratio as lcs_ratio
import networkx as nx


class CompaniesEntityResolution:
    def __init__(self, companies_data: pd.DataFrame, key_fields: list = None):
        self.companies_data = companies_data
        self.key_fields = key_fields or []
    

    def create_blocks(self):
        """
        Create blocks of similar records based on the key fields.
        """

        # 1. web domain-based blocking
        self.companies_data['domain_block'] = self.companies_data['website_domain'].fillna("")

        # 2. Name-based blocking using first company name
        self.companies_data['name_block'] = self.companies_data['company_name'].fillna("")

        # 3. Phone number-based blocking
        self.companies_data['phone_block'] = self.companies_data['phone_numbers'].fillna("")
        
        return self
    

    def generate_candidate_pairs(self):
        """
        Generate candidate pairs for comparison using multiple blocking strategies.
        
        Returns:
            list: List of candidate pairs (tuples of record indices) for comparison
        """
        
        if 'domain_block' not in self.companies_data.columns:
            self.create_blocks()
        
        # Track all candidate pairs across blocking strategies
        candidate_pairs = set()
        total_blocks = 0

        for block_column in ['domain_block', 'phone_block', 'name_block']:
            # Skip empty blocks
            blocks = self.companies_data[self.companies_data[block_column].str.len() > 0]
            block_groups = blocks.groupby(block_column)
            
            # Process each block
            for _, indices in block_groups.groups.items():
                block_size = len(indices)
                
                total_blocks += 1

                if block_size > 1:  # Only generate pairs if there are at least 2 records
                    block_pairs = set(combinations(indices, 2))
                    candidate_pairs.update(block_pairs)
        
        # Convert to list of sorted pairs to ensure consistent ordering
        sorted_pairs = [(min(i, j), max(i, j)) for i, j in candidate_pairs]
        
        print(f"domain_block: {self.companies_data['domain_block'].nunique()} unique blocks")
        print(f"phone_block: {self.companies_data['phone_block'].nunique()} unique blocks")
        print(f"name_block: {self.companies_data['name_block'].nunique()} unique blocks")

        return sorted_pairs
    

    def compare_candidate_pairs(self, pairs: list[tuple]) -> pd.DataFrame:
        """
        Compare candidate pairs and generate comprehensive similarity features
        (name, domain, phone, location, industry).

        Parameters:
            candidate_pairs (list): list of (idx1, idx2) pairs.

        Returns:
            pd.DataFrame: DataFrame with many similarity features for each candidate pair.
        """   

        results: list[dict] = [] # Will contain the candidates indexes along with similarity scores
        for idx1, idx2 in pairs:
            row1 = self.companies_data.loc[idx1]
            row2 = self.companies_data.loc[idx2]

            # --- 1) NAME SIMILARITY (Longest Common Subsequence) ---
            name_sim = self._compute_lcs_similarity(row1['company_name'], row2['company_name'])

            # --- 2) DOMAIN SIMILARITY (Exact match or partial) ---
            domain_sim = self._compute_domain_similarity(row1['website_domain'], row2['website_domain'])

            # --- 3) PHONE SIMILARITY (Normalize and compare) ---
            phone_sim = self._compute_phone_similarity(row1['phone_numbers'], row2['phone_numbers'])

            # --- 4) LOCATION SIMILARITY (Composite weighted similarity) ---
            location_sim = self._compute_location_similarity(row1, row2)

            # --- 5) INDUSTRY SIMILARITY (Jaro-Winkler) ---
            industry_sim = self._compute_jaro_winkler(row1.get('main_industry', ''), 
                                                 row2.get('main_industry', ''))
            
            results.append({
                "idx1": idx1,
                "idx2": idx2,
                "name_similarity": name_sim,
                "domain_similarity": domain_sim,
                "phone_similarity": phone_sim,
                "location_similarity": location_sim,
                "industry_similarity": industry_sim
            })

        return pd.DataFrame(results)


    def _compute_jaro_winkler(self, text1: str, text2: str):
        """
        Returns:
            Jaro-Winkler similarity for two text fields.
        """
        if not text1 or not text2:
            return 0.0
        
        return jaro_winkler_similarity(text1, text2)
    
    
    def _compute_lcs_similarity(self, text1: str, text2: str) -> float:
        """
        Compute the Longest Common Subsequence (LCS) similarity for two text fields.
        
        Parameters:
            text1 (str): First text field
            text2 (str): Second text field
        
        Returns:
            float: LCS similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        return lcs_ratio(text1, text2)


    def _compute_phone_similarity(self, phone1: str, phone2: str):
        """
        Returns:
            1 if the phone numbers match, 0 otherwise
        """
        if not phone1 or not phone2:
            return 0.0
        return float(phone1 == phone2)
    

    def _compute_domain_similarity(self, domain1: str, domain2: str):
        return self._compute_jaro_winkler(domain1, domain2) if domain1 and domain2 else 0.0
    
    
    def _compute_location_similarity(self, loc1, loc2):
        """
        Returns:
            similarity for location, weighting location attributes.
        """

        weights = {
            'main_country': 0.1,
            'main_region':  0.1,
            'main_city':    0.2,
            'main_street': 0.2,
            "main_street_number": 0.2,
            "main_postcode": 0.2
        }

        total_score = 0.0
        for field, weight in weights.items():
            val1 = loc1.get(field, "")
            val2 = loc2.get(field, "")
            if not val1 or not val2:
                continue
            total_score += self._compute_jaro_winkler(val1, val2) *  weight
        
        return total_score
    

    def classify_pairs(self, similarity_df: pd.DataFrame, domain_threshold=0.9, name_threshold=0.5) -> pd.DataFrame:
        """
        Classify candidate pairs as matches using a rule-based approach that handles missing data.
        
        Parameters:
            similarity_df (pd.DataFrame): DataFrame with similarity features
            domain_threshold (float): Domain similarity threshold
            name_threshold (float): Name similarity threshold
            phone_threshold (float): Phone similarity threshold
            
        Returns:
            pd.DataFrame: Original DataFrame with 'is_match' and 'match_reason' columns
        """

        # Initialize match status
        similarity_df['is_match'] = False

        # Rule 1: High domain similarity AND approximately same name
        domain_name_matches = (
            (similarity_df['domain_similarity'] >= domain_threshold) &
            (similarity_df['name_similarity'] >= name_threshold)
        )

        # Rule 2: Exact match on primary phone, and approximately same name
        phone_name_matches = (
            (similarity_df['phone_similarity'] == 1.0) &
            (similarity_df['name_similarity'] >= name_threshold)
        )

        # Rule 3: Perfect name match and similar location
        name_location_matches = (
            (similarity_df['name_similarity'] >= 0.95) &
            (
                (similarity_df['location_similarity'] >= 0.3) |
                (similarity_df['industry_similarity'] >= 0.5)
            )
        )

        # Rule 4: Similar domain with exact same phone
        domain_phone_matches = (
            (similarity_df['domain_similarity'] >= domain_threshold) &
            (similarity_df['phone_similarity'] == 1.0)
        )

        # Rule 5: Similar location and one of the domain/phone
        location_matches = (
            (similarity_df['location_similarity'] >= 0.90) &
            (
                (similarity_df['phone_similarity'] == 1.0) |
                (similarity_df['domain_similarity'] >= domain_threshold)
            )
        )

        similarity_df.loc[domain_name_matches, 'is_match'] = True

        similarity_df.loc[phone_name_matches, 'is_match'] = True

        similarity_df.loc[name_location_matches, 'is_match'] = True

        similarity_df.loc[domain_phone_matches, 'is_match'] = True

        similarity_df.loc[location_matches, 'is_match'] = True

        return similarity_df
    

    def create_company_clusters(self, matched_pairs_df: pd.DataFrame) -> dict:
        """
        Group matching records into clusters representing unique companies.
        
        Parameters:
            matched_pairs_df (pd.DataFrame): DataFrame with classified pairs
            
        Returns:
            dict: Mapping of original indices to cluster IDs
        """
        matches = matched_pairs_df[matched_pairs_df['is_match']]
        
        # Nodes are company records and edges represent matches
        G = nx.Graph()

        for _, row in matches.iterrows():
            G.add_edge(row['idx1'], row['idx2'])
        
        # Convert connected components to list
        # *clusters* is a list of sets, each set is a cluster of indices
        clusters = list(nx.connected_components(G))

        index_to_cluster = {}
        for cluster_id, indices in enumerate(clusters):
            for idx in indices:
                index_to_cluster[idx] = cluster_id
        
        all_indices = set(self.companies_data.index)
        clustered_indices = set(index_to_cluster.keys())
        next_cluster_id = len(clusters)

        for idx in all_indices - clustered_indices:
            index_to_cluster[idx] = next_cluster_id
            next_cluster_id += 1
        
        return index_to_cluster


    def create_deduplicated_dataset(self, df, cluster_column='cluster_id') -> pd.DataFrame:
        """
        Create a deduplicated dataset with one representative record per cluster.
        
        Parameters:
            df (pd.DataFrame): Original dataframe with cluster IDs
            cluster_column (str): Name of the column containing cluster IDs
            
        Returns:
            pd.DataFrame: Deduplicated dataframe with one record per cluster
        """
        
        # all unique cluster IDs
        cluster_ids = df[cluster_column].unique()
        
        # list for representative records
        representatives = []
        
        # Process clusters in batches for better performance
        for cluster_id in cluster_ids:
            cluster_records = df[df[cluster_column] == cluster_id]
            representative = self._select_representative_record(cluster_records)
            representatives.append(representative)
        
        # Create dataframe from all representatives at once
        unique_companies = pd.DataFrame(representatives)
        
        return unique_companies


    def _select_representative_record(self, cluster_records):
        """
        Select the best representative record from a cluster.
        
        Parameters:
            cluster_records (pd.DataFrame): All records in a cluster
            
        Returns:
            Series: The representative record
        """
    
        if len(cluster_records) == 1:
            return cluster_records.iloc[0]
        
        # Calculate completeness scores (vectorized)
        completeness_scores = cluster_records.notna().sum(axis=1)
        
        # If there is only one record with max completeness, return it
        max_score = completeness_scores.max()
        max_indices = completeness_scores[completeness_scores == max_score].index
        
        if len(max_indices) == 1:
            return cluster_records.loc[max_indices[0]]
        
        # Filter to only the records with maximum completeness score
        candidates = cluster_records.loc[max_indices]
        
        # Create a priority score for tie-breaking (domain = 2 points, name = 1 point)
        priority_score = (
            candidates['website_domain'].notna().astype(int) * 2 + 
            candidates['company_name'].notna().astype(int)
        )
        
        # Get the index with highest priority
        if priority_score.max() > 0:
            best_idx = priority_score.idxmax()
            return cluster_records.loc[best_idx]
        
        # If still tied, return the first one
        return cluster_records.loc[max_indices[0]]


    def save_cluster_mappings(self, index_to_cluster: dict, output_path: str = "company_clusters.json"):
        """
        Saves the cluster mappings to a JSON file, with cluster IDs and their corresponding record indices.

        Parameters:
            index_to_cluster (dict): Mapping of original indices to cluster IDs.
            output_path (str): Path to save the JSON file.
        """
        import json
        import os

        # Reorganize by cluster_id for better readability
        clusters_to_indices = {}
        for idx, cluster_id in index_to_cluster.items():
            if cluster_id not in clusters_to_indices:
                clusters_to_indices[cluster_id] = []
            clusters_to_indices[cluster_id].append(int(idx))

        # Sort clusters by size (largest first)
        sorted_clusters = {
            k: v for k, v in sorted(
                clusters_to_indices.items(),
                key=lambda item: len(item[1]),
                reverse=True
            )
        }

        readable_clusters = {
            f"cluster_{k}_{len(v)}_records": v
            for k, v in sorted_clusters.items()
        }

        with open(output_path, 'w') as f:
            json.dump(readable_clusters, f, indent=2)

        print(f"Saved cluster mappings to {os.path.abspath(output_path)}")
