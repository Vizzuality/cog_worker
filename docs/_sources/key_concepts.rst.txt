Key concepts
============

Leverage GeoTIFF overviews
--------------------------
Cloud Optimized Geotiffs are structured to enable fast reads of subsets by internally tiling data, and including downsampled versions of themselves, called "overviwes" or "pyramids".

 - When conducting analyses at resolutions lower than that of the input data, cog_worker reads from the internal overviews in COGs.
 - (!) The resampling method used to produce input dataset overviews will affect your analysis results.


Compute outputs only
--------------------
Rather than starting from the complete source datasets and working forward, cog_worker only reads the data necessary to compute the output.

 - Analyses are defined agnostic to projection and scale, so the same function can be used to render a web tile map as to produce a high resolution GeoTIFF in a custom projection.
 - Inputs are automatically resampled, reprojected, and clipped to the output scale and projection on read. 
 - Low resolution previews can be generated quickly.
 
Chunk large analysis
--------------------
cog_worker has functions to chunk analyses into managable pieces (e.g. 512x512 tiles), so you can work more effectively at scale.

- Tiles are computed with a buffer to account for edge effects.
