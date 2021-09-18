# FACE SWAPPING
## Steps:
1. Detect faces (apply for the biggest face was detected)
2. Find 68 facial landmarks points
	- Find convex hull from 68 points.
	- Find convex hull and inner mouth points 
3. Delaunay triangles for both faces:
	- Using 68 points.
	- Using only convex hull points and its indexes
	- Using only convex hull and inner mouth points and its indexes
4. Warp triangles:
	- Get triangles indexes from both face and warp with corresponding indexes to each other.
	- Create new face with the same shape to the destination face.
5. Swap face:
	- Apply new face into the destination image.
	- Using seamless clone to balance the color between the new and the destination image.
	
# Libraries and Modules supported:
1. OpenCV2: 
	- Read images
	- Convert color base
	- Detect faces 
	- Detect facial landmarks
	- Warp Triangles
	- Seamless clone image
	- Show image
	
2. Numpy:
	- Support process indexes and points 
	- Create mask
	- etc

3. Streamlit:
	- Create an application 
	