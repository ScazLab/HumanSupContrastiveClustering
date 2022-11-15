 import numpy as np
 
 
def overlap(box1, box2):
        try:
            intersect_area = float(box1.intersection(box2).area) 
            if intersect_area == 0.0:
                return 0.0
            return intersect_area / (box1.area + box2.area - intersect_area)
        except Exception as e:
            print(box1.area, box2.area, intersect_area)
            print(e)
            print(box1, box2)
            return 0.0

def main():
    op = overlap([], [])

if __name__ == "__main__":
    main()