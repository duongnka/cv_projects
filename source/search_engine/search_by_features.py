from search_engine.search_utils import *
from scipy import spatial


class SearchEngineByFeature:
    def __init__(self) -> None:
        self.load_data()

    def load_data(self):
        self.inverted_features_index = load_obj('inverted_features_index')
        self.feature_extractor = Resnet18SearchEngine()
        self.list_images = load_obj('list_images')
        self.images_features = load_obj('image_features_resnet18')
    
    def extract_query_feature(self, query_image):
        query_feature = self.feature_extractor.feature_extractor(query_image)
        return query_feature
    
    def get_related_images_ids(self, query_feature):
        result_ids = []
        for feature, value in enumerate(query_feature):
            if (feature not in self.inverted_features_index 
                or value == 0 
                or value not in self.inverted_features_index[feature]):
                continue
            
            ids = self.inverted_features_index[feature][value]
            result_ids = get_combined_list(result_ids, ids)
        return result_ids
    
    def calculate_distance(self, query_feature, result_ids):
        results = []
        for id in result_ids:
            d = 1 - spatial.distance.cosine(self.images_features[id], query_feature)
            results[id] = d
        return sorted([(v, k) for (k, v) in results.items()],reverse=True)
    
    def search(self, query_image):
        query_feature = self.extract_query_feature(query_image)
        result_ids = self.get_related_images_ids(query_feature)
        results = self.calculate_distance(query_feature, result_ids)
        return results
    