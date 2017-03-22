// draw2 is like a 'sandbox' to test out different drawing approaches
// e.g., a first goal is to write a spline <-> raster image 
// transformer that is differentiable
//
// jefan 3/22/17

// from Daniel Ritchie
// Just write all the spline evaluation functions in AD code. 
// Then, when sampling the spline onto the
// image grid, rather than using inside/outside tests 
// (i.e. is this pixel inside the width of the spline being rendered),
// use some continuous function of the distance to the spline 
// boundary with controllable falloff (e.g. some sort of exponential). 
// If you're doing this in adnn, you'd probably want to write 
// the sampling function as a new adnn primitive 
// (to handle derivatives through pixels efficiently).


function DrawObject(width, height, visible){
  this.canvas = $('<canvas/>', {
    "class": "drawCanvas",
    "Width": width + "px",
    "Height": height + "px"
  })[0];
  if (visible==true){
    $(this.canvas).css({"display": "inline"});
    var container = wpEditor.makeResultContainer();
    $(container).append(this.canvas);
  };
  this.paper = new paper.PaperScope();
  this.paper.setup(this.canvas);
  this.paper.view.viewSize = new this.paper.Size(width, height);
  this.redraw();
}

DrawObject.prototype.newPath = function(strokeWidth, opacity, color){
  var path = new this.paper.Path();
  path.strokeColor = color || 'black';
  path.strokeWidth = strokeWidth || 4;
  path.opacity = opacity || 0.8;
  return path;
};

DrawObject.prototype.newPoint = function(x, y){
  return new this.paper.Point(x, y);
};

DrawObject.prototype.newCurve = function(point1, point2, point3, point4) {
  return new this.paper.Curve(point1, point2, point3, point4);
};

DrawObject.prototype.circle = function(x, y, radius, stroke, fill){
  var point = this.newPoint(x, y);
  var circle = new this.paper.Path.Circle(point, radius || 50);
  circle.fillColor = fill || 'black';
  circle.strokeColor = stroke || 'black';
  this.redraw();
};

DrawObject.prototype.polygon = function(x, y, n, radius, stroke, fill){
  var point = this.newPoint(x, y);
  var polygon = new this.paper.Path.RegularPolygon(point, n, radius || 20);
  polygon.fillColor = fill || 'white';
  polygon.strokeColor = stroke || 'black';
  polygon.strokeWidth = 4;
  this.redraw();
};

DrawObject.prototype.line = function(x1, y1, x2, y2, strokeWidth, opacity, color){
  var path = this.newPath(strokeWidth, opacity, color);
  path.moveTo(x1, y1);
  path.lineTo(this.newPoint(x2, y2));
  this.redraw();
};

DrawObject.prototype.drawSpline = function(startX, startY, midX, midY, endX, endY){
  var path = this.newPath();
  path.strokeColor = 'black';
  path.add(this.newPoint(startX, startY));
  path.add(this.newPoint(midX, midY));
  path.add(this.newPoint(endX, endY));

  path.smooth();

  this.redraw();
};

DrawObject.prototype.redraw = function(){
  this.paper.view.draw();
};

DrawObject.prototype.toArray = function(){
  var context = this.canvas.getContext('2d');
  var imgData = context.getImageData(0, 0, this.canvas.width, this.canvas.height);
  return imgData.data;
};

DrawObject.prototype.distanceF = function(f, cmpDrawObject){
  if (!((this.canvas.width == cmpDrawObject.canvas.width) &&
        (this.canvas.height == cmpDrawObject.canvas.height))){
    console.log(this.canvas.width, cmpDrawObject.canvas.width,
                this.canvas.height, cmpDrawObject.canvas.height);
    throw new Error("Dimensions must match for distance computation!");
  }
  var thisImgData = this.toArray();
  var cmpImgData = cmpDrawObject.toArray();
  return f(thisImgData, cmpImgData);
};

DrawObject.prototype.distance = function(cmpDrawObject){
  var df = function(thisImgData, cmpImgData) {
    var distance = 0;
    for (var i=0; i<thisImgData.length; i+=4) {
      var col1 = [thisImgData[i], thisImgData[i+1], thisImgData[i+2], thisImgData[i+3]];
      var col2 = [cmpImgData[i], cmpImgData[i+1], cmpImgData[i+2], cmpImgData[i+3]];
      distance += euclideanDistance(col1, col2);
    };
    return distance;
  };
  return this.distanceF(df, cmpDrawObject)
};

DrawObject.prototype.destroy = function(){
  this.paper = undefined;
  $(this.canvas).remove();
}

function Draw(s, k, a, width, height, visible){
  return k(s, new DrawObject(width, height, visible));
}

function loadImage(s, k, a, drawObject, url){
  // Synchronous loading - only continue with computation once image is loaded
  var context = drawObject.canvas.getContext('2d');
  var imageObj = new Image();
  imageObj.onload = function() {
    var raster = new drawObject.paper.Raster(imageObj);
    raster.position = drawObject.paper.view.center;
    drawObject.redraw();
    resumeTrampoline(function() { return k(s) });
  };
  imageObj.src = url;
}
