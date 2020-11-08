# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.





import pandas as pd

films = { 'title': [1,2,3,4,5], 'rank': [10,20,30,40,50], 'synopsis': [100,200,300,400,500], 'cluster': [1000,2000,3000,4000,5000], 'genre': [10000,20000,30000,40000,50000] }

frame = pd.DataFrame(films, index = ['a','b','c','d','e'] , columns = ['rank', 'title', 'cluster', 'genre'])

print(frame)

group = frame['rank'].groupby(frame['cluster']).mean()

print(group)

group = frame.groupby(frame['cluster']).mean()

print(group)