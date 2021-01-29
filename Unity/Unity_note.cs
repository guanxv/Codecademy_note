// shortcut key

F focus on itme
R scale




unity unit : default 1 unit = 1 meter

void update() //runs  every begining of frame

void fixed_update() //runs when the pysics changes, this is where the physics code to go.

void LateUpdate() // runs at every end of frame, good for camera position update.


// ------------------------ material ---------------------------------

fix for pink purple issue with material.

Edit ==> Render Pipeline ==> Universal Render PipeLine => update project material...


// any game object  with collider and rigidbody is cosider as dynamic.  other component is considered static. 
if a static object is moving , it will requir more resources for unity to calculate each farme. 

in the rigidbody, there is Use Gravity. 
is Kinematic, a kinematic is not affected by forces. but still do animate.