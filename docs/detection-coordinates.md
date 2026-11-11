# Detection coordinates and tracking compatibility

OTVision's YOLO adapter converts YOLO's center-based `xywh` into
`x = center_x - w / 2` and `y = center_y - h / 2`. The persisted
`x, y, w, h` values in `.otdet` and `.ottrk` therefore describe the
**top-left bounding-box corner** and its width and height, in image pixels
(or normalized image coordinates when that export option is used).

The IoU tracker reconstructs a box from these four values as
`(x, y, x + w, y + h)`. It does not change the serialized coordinate
values. Existing `.otdet` files can be retracked without running detection
again. Existing `.ottrk` files remain readable with the same pixel
coordinates and track IDs; retracking may produce different track IDs
because corrected box overlaps change association decisions. Preserve old
results as separate artifacts when comparing counts or analytics exports.

The transformation step currently projects the stored `x, y` coordinate,
which is the bounding box's top-left corner, to the GeoPackage. This is
**not** an observed contact point on the road. A ground-contact anchor,
validation of reference points and explicit output CRS handling are
separate work; this tracker correction does not claim to fix georeferenced
positions or derived speeds.
