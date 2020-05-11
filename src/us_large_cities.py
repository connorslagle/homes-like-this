class USMetros():
    '''
    This class loads US cities based on size/growth rate threshold. 
    '''
    def __init__(self, selection_criteria='size', size_threshold=500000, rate_threshold=20):
        self.criteria = selection_criteria
        if selection_criteria == 'size':
            self.threshold = size_threshold
        else:
            self.threshold = rate_threshold
        
        # city data dict: key = city name, value= [pop (milions) (2018), growth rate (%), 2010-2018] - src: wikipedia
        self.city_data = {'New York': [8.4, 2.8], 'Los Angeles':[3.4, 5.22]}
    
    def _populate_cities(self):


'''
Gonna webscrape too big to manual enter

state id:               //table[@class='wikitable sortable jquery-tablesorter'][1]/tbody/tr/text()
city:
state:
pop (2018):
pop (2010):
change(2010-18):
land area (2016):
pop density (2016):
Location (Lat/Long):
'''