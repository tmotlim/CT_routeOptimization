# <G6T16>
# <Wang Zi,Lee Hyeonjeong,Lim Zhong Zhen Timothy>

from utility import *
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

# The total distance of the route. Including the distance from origin (edited to be based on distance/speed)
def cost(route):
    route_distance = euclidean_distance([0,0], coord_map[route[0]])
    for i in range(1,len(route)):
        route_distance += euclidean_distance(coord_map[route[i - 1]], coord_map[route[i]])
    route_distance += euclidean_distance(coord_map[route[len(route) - 1]], [0,0])
    return route_distance

# Function to calculate Euclidean Distance
def euclidean_distance (x, y):
    time_taken = (((float(x[0])-float(y[0]))**2 + (float(x[1])-float(y[1]))**2)**0.5)/truck_spd
    return time_taken	

#Function to calculate plane time
def euclidean_time_plane (x, y):
    time_distancea = (((float(x[1])-float(y[1]))**2 + (float(x[2])-float(y[2]))**2)**0.5)/plane_spd
    return time_distancea
def euclidean_time_truck (x, y):
    #print ('truck '+ str(type(x[2])),str(type(y[2])))
    time_distancea = (((float(x[1])-float(y[1]))**2 + (float(x[2])-float(y[2]))**2)**0.5)/truck_spd
    return time_distancea

#eucl
def euclidean_distance2 (x1,y1,x2,y2):
    #print('order :'+ str(x1) + str(y1)+ 'airport :' +str(x2),str(y2))
    time_taken = (((x1-x2)**2 + (y1-y2)**2)**0.5)/truck_spd
    return time_taken


def all_dict_generation(all_list):
    location_dict = {}
    for the_list in all_list:
        for item in the_list:
            if len(item) < 4:
                location_dict[item[0]] = [item[0], float(item[1]), float(item[2])]
            else:
                location_dict[item[0]] = [item[0], float(item[1]), float(item[2]),float(item[3])]
    return location_dict

# Schedule function
def schedule_q2(orders, airports, truck_speed, plane_speed, number_trucks):
    location_dict = all_dict_generation([orders, airports])
    global truck_spd
    truck_spd = truck_speed
    global plane_spd
    plane_spd = plane_speed

    #get order to closest airport into dictionary
    Ordertoclosest_airport = {}
    for order in orders:
        closest_airport = ''
        closest_airport_distance = 10000000000
        for airport in airports:
            distance = euclidean_distance2(float(order[1]),float(order[2]),float(airport[1]),float(airport[2]))
            #print("dist: " + str(distance))
            if  distance < closest_airport_distance:
                closest_airport = airport[0]
                closest_airport_distance = distance

        
        Ordertoclosest_airport[order[0]] = closest_airport
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
            #result[i] = two_opt(result[i])
            #print ('best_truck: '+ str(i))
            
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
        
    # Due to randomness of KMeans, we repeat the process 1 times and stores the best (edited)
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
        #print ('Running'+ str(i))
        #either insert airports here, Safe insertion here
    insertion_route = []
    insertion_index = []
    insertion_airports = []
    for i in range(len(final_res)):
        for j in range(len(final_res[i])-1):
            curr_order = final_res[i][j]
            next_order = final_res[i][j+1]
            curr_order_loc = location_dict[curr_order]
            airport_loc = location_dict[Ordertoclosest_airport[curr_order]]
            next_airport_loc = location_dict[Ordertoclosest_airport[next_order]]
            next_order_loc = location_dict[next_order]
            airport_takeoff_limit = airport_loc[3] 
            if airport_loc != next_airport_loc and airport_takeoff_limit > 0:
                if (euclidean_time_truck(curr_order_loc, airport_loc) + euclidean_time_truck(next_airport_loc,next_order_loc)) < euclidean_time_truck(curr_order_loc,next_order_loc):
                     
                    #print('INSERT: '+ final_res[i][j] + " : " + Ordertoclosest_airport[curr_order] + " : " + Ordertoclosest_airport[next_order] )
                    insertion_index.append(j)
                    insertion_index.append(j)
                    insertion_route.append(i)
                    insertion_route.append(i)
                    insertion_airports.append(Ordertoclosest_airport[next_order])
                    insertion_airports.append(Ordertoclosest_airport[curr_order])
                    location_dict[Ordertoclosest_airport[curr_order]][3] -= 1
                    
                    #newairport_loc[3] -= 1 

                
    if len(insertion_index) != 0:
        for z in range(len(insertion_index)): #0,1,2,3,4,5,6
            insert_at = insertion_index[z]+ 1 #10,10,15,15,20,20
            final_res[insertion_route[z]].insert(insert_at+1,insertion_airports[z])
     
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

#how to tag airport and order? it?