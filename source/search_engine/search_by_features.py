from search_engine.search_utils import *
# from search_utils import *
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
            image_path = self.list_images[id]
            k = image_path[image_path.rfind("/") + 1:]
            d = spatial.distance.cosine(self.images_features[k], query_feature)
            results.append({ 'index' :id, 'dist': d})

        return sorted(results, key=lambda k: (k['dist']))
    
    def load_result_images(self, results):
        images_links = []
        for index in range(25):
            i = results[index]['index']
            path = './search_engine' + self.list_images[i].replace('\\', '/')[1:]
            images_links.append(path)
        return images_links

    def search(self, query_image):
        query_feature = self.extract_query_feature(query_image)
        result_ids = self.get_related_images_ids(query_feature)
        results = self.calculate_distance(query_feature, result_ids)
        return self.load_result_images(results)


# path = './data_set/17flowers/image_0001.jpg'
# query_img = read_img_PIL(path)
# engine = SearchEngineByFeature()
# results = engine.search(query_img)
# print(results[0])