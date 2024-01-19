Tried to run hydra with gym pybullet drones

gpd is in general slower than what we need for hydra

hydra requires good res semantic pointclouds being pumped into it at high speeds.

gpd at it's best could do 640x480 res segmentation image without gui at 2Hz which is too slow (maybe). The main issue was that hydra would create a triangle mesh initially but just stop after that
moreover, depth image proc also wouldnt work at low resolutions which i find weird.
wrote a custom publisher for rgb pointclouds which was more reliable than depth image proc but hydra doesnt work with that even

to make gpd faster i had to
- disable gui
- use the cudagl docker image
    - this helped me run non gui mode with EGL renderer. if the cudagl image wasnt used, this resulted in GL_RENDERER=llvmpipe.
- smaller images
- call getdroneimages outside the main classes without recording bs


maybe disabling textures and just having colors may help. we're using gt semantics anyways
also could be helpful to use primitives for the rooms and stuff. instead of meshes