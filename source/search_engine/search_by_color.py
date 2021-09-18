from numpy.lib.histograms import histogram
from search_engine.search_utils import *

class HistogramSearch:
    
    def __init__(self, 
    name = 'inverted_histogram_index', 
    bins = [4,4,4], 
    list_images = 'list_images',
    images_histograms = 'images_histograms') -> None:

        self.inverted_histogram_index = load_obj(name)
        self.feature_extractor = RGBHistogram(bins)
        self.list_images = load_obj(list_images)
        self.images_histograms = load_obj(images_histograms)
    
    def extract_query_feature(self, query_image):
        query_histogram = self.feature_extractor.describe(query_image)
        query_histogram = [round(digit) for digit in query_histogram * 10e2]
        return query_histogram    

    def get_related_images_ids(self, query_histogram):
        result_ids = []
        for feature, value in enumerate(query_histogram):
            if (feature not in self.inverted_histogram_index 
                or value == 0 
                or value not in self.inverted_histogram_index[feature]):
                continue
            
            ids = self.inverted_histogram_index[feature][value]
            result_ids = get_combined_list(result_ids, ids)
        return result_ids

    def calculate_distance(self, query_histogram, result_ids):
        results = []
        for i in result_ids:
            image_path = self.list_images[i]
            k = image_path[image_path.rfind("/") + 1:]
            histogram = self.images_histograms[k]
            distance = chi2_distance(query_histogram, histogram)
            results.append({ 'index' :i, 'dist': distance})
        return sorted(results, key=lambda k: (k['dist']))

    def load_result_images(self, results):
        result_images = []
        for index in range(10):
            i = results[index]['index']
            path = './search_engine' + self.list_images[i].replace('\\', '/')[1:]
            image = cv2.imread(path)
            result_images.append(image)
        print(len(result_images))
        return result_images

    def search(self, query_image):
        query_histogram = self.extract_query_feature(query_image)
        result_ids = self.get_related_images_ids(query_histogram)
        results = self.calculate_distance(query_histogram, result_ids)
        return self.load_result_images(results)
    