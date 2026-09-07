# Dhan capture-gate proxy

This filter approximates the capture constraints documented in
`docs/ML_RESEARCH_ARCHITECTURE.md`. It is not claimed to be byte-for-byte
equivalent to the product browser implementation, which is maintained in the
separate product repository.

## Proxy v1

- Apply EXIF orientation.
- Resize to a 224×224 analysis canvas.
- Blur score: variance of the grayscale Laplacian; require at least 80.
- Leaf coverage: HSV green-mask fraction; require at least 60%.
- OpenCV HSV mask: hue 20–100, saturation at least 40, value at least 40.

The filter records every score and rejection reason in `scores.csv`. Thresholds
and mask parameters must be synchronized with the actual product before this
proxy is used as a production acceptance claim.

## Result

Of 606 compatible Dhan images, 172 passed both gates. All eligible images came
from the field-background subset: 73 blast, 48 brown spot, and 51 tungro.
The exp16 checkpoint reached 0.3081 full-model accuracy and 0.3314
supported-class macro-F1 on this eligible subset, confirming that capture
filtering alone does not close the observed domain gap.
