import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import random
import copy

coord_map = {}

# Helper function which swaps edges between two cities
def swap_2opt(route, i, k):
	assert i >= 0 and i < (len(route) - 1)
	assert k > i and k < len(route)
	new_route = route[0:i]
	new_route.extend(reversed(route[i:k + 1]))
	new_route.extend(route[k+1:])
	assert len(new_route) == len(route)
	return new_route

# Two Optimization Algorithm to swap two edges until we find local minimum
def two_opt(route):
    improved = True
    lowest_cost = cost(route)
    while improved: 
        improved = False
        for i in range(len(route) - 1): 
            for j in range(i + 1, len(route)):
                new_route = swap_2opt(route, i, j)
                new_cost = cost(new_route)
                if new_cost < lowest_cost:
                    lowest_cost = new_cost
                    route = new_route 
                    improved = True 
    return route

# The total distance of the route. Including the distance from origin
def cost(route):
    route_distance = euclidean_distance([0,0], coord_map[route[0]])
    for i in range(1,len(route)):
        route_distance += euclidean_distance(coord_map[route[i - 1]], coord_map[route[i]])
    route_distance += euclidean_distance(coord_map[route[len(route) - 1]], [0,0])
    return route_distance

# Function to calculate Euclidean Distance
def euclidean_distance (x, y):
    distance = ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
    return distance	

# Schedule function
def schedule_q1(orders, number_trucks):
    # Converts String into Float and creates the coordinate map
    for row in orders:
        row[1] = float(row[1])
        row[2] = float(row[2])
        coord_map[row[0]] = [row[1], row[2]]

    coord_map['origin'] = [0,0]
    ords = pd.DataFrame.from_records(orders, columns=['order_id', 'x', 'y'])

    orders_no_id = ords[['x','y']]
    glob_min = 100000000000000000000000000000
    final_res = []

    # Limits the number of clusters created
    if number_trucks <=5:
        limit = 0
    else:
        limit = number_trucks//2

    # Builds the cluster dataframe, which aids with KMeans Algorithm
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = orders_no_id.index.values
    cluster_map['x'] = ords['x']
    cluster_map['y'] = ords['y']
    cluster_map['order_id'] = ords['order_id']

    best_truck = 0
    # Runs the K Means from Number of Truck times to limit times.
    while number_trucks > limit:
        km = MiniBatchKMeans(n_clusters=number_trucks)
        groups = km.fit(orders_no_id)


        cluster_map['cluster'] = groups.labels_
        result = [] 
        local_max = 0

        # Builds the route for each cluster
        for i in range(number_trucks):
            # gets the nearest neighbour route for each cluster
            result.append(randomized_nearest_neighbour(cluster_map, i))
            result[i] = two_opt(result[i])
            c = cost(result[i])
            if(c > local_max):
                local_max = c

        # If the local max score better than the global version, we store the current one as best
        if(local_max < glob_min):
            glob_min = local_max
            final_res = result
            best_truck = number_trucks

        number_trucks -= 1        

    if best_truck == 1:
        return final_res
        
    # Due to randomness of KMeans, we repeat the process 20 times and stores the best
    for i in range(0, 21):
        r = random.randrange(0, 51)
        km = MiniBatchKMeans(n_clusters=best_truck, random_state=r)
        groups = km.fit(orders_no_id)
        cluster_map['cluster'] = groups.labels_
        result = [] 
        local_max = 0
        for j in range(best_truck):
            result.append(randomized_nearest_neighbour(cluster_map, j))
            result[j] = two_opt(result[j])
            c = cost(result[j])
        
            if(c > local_max):
                local_max = c

        if(local_max < glob_min):
            glob_min = local_max
            final_res = result
        
    return final_res

# Gets the minimum distance city to the current city
def get_min(city, cities, threshold):
  def my_cmp(c):
    return euclidean_distance(coord_map[city], coord_map[c])
  sorted_cities = sorted(cities, key=my_cmp)
  return sorted_cities[0:threshold]

# Get nearest neighbour route.
def randomized_nearest_neighbour(df, cluster_index, threshold=None):
    if threshold is None:
        threshold = 1
    
    curr = "origin"
    cluster_cities = copy.deepcopy(list(df[df['cluster']==cluster_index]['order_id'].values))
    length = len(cluster_cities)
    route = []

    while length > 0:
        min_list = get_min(curr, cluster_cities, threshold)
        rand = random.randrange(0, len(min_list))

        next_city = min_list[rand]
        route.append(next_city)

        cluster_cities.remove(next_city)
        curr = next_city
        length -= 1

    return route