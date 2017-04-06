// sketcher.js 
// differentiable spline renderer, built from adnn + paper
// jefan 4/4/17


// require dependencies
var ad = require('adnn/ad');
var paper = require('paper');
var catmullFitter = require('catmullFitter');
_ = require('underscore');
require('numeric');

// distance between two points (x,y)
function dist(x1, y1, x2, y2) {
  var xdiff = ad.scalar.sub(x1, x2);
  var ydiff = ad.scalar.sub(y1, y2);
  return ad.scalar.sqrt(ad.scalar.add(
    ad.scalar.mul(xdiff, xdiff),
    ad.scalar.mul(ydiff, ydiff)
  ));
}

// eG = evaluateGaussian
// evaluate Gaussian loss function, normalized to 1 at distance t-m = 0
// t = distance from mean
// m = mean
// s = standard deviation
function eG(t,m,s) {
	a = ad.scalar.div(1,(ad.scalar.mul(s,Math.sqrt(ad.scalar.mul(2,Math.PI)))));
	b = Math.exp( -Math.pow(m-t,2)/(ad.scalar.mul(2,Math.pow(s,2))));
	c = ad.scalar.mul(a,b);
	return c;
}

var doop = eG(1,0,1);
console.log(doop);

// gS = genStops
// generate gradient stops for normal vector to main path
// t = distance from main path
// s = standard deviation of gaussian 
function gS(t,s) { 
    n = eG(0,0,s);
    c = 1 - ad.scalar.div(eG(t,0,s),n);
    var s = new Color(c,c,c);    
    return s;
}


// generate generic cubic 
// y = ax^3 + bx^2 + cx + d 
function evalCubic(_x,a,b,c,d) {		
	_y = ad.scalar.add(
			ad.scalar.add(
				ad.scalar.add(
					ad.scalar.mul(a,ad.scalar.pow(_x,3)),
					ad.scalar.mul(b,ad.scalar.pow(_x,2))
				),
			 	ad.scalar.mul(c,_x)
			),d
		 );
	return _y;
}

function genCubic(x0,x1,N,a,b,c,d) {
	xvals = numeric.linspace(x0,x1,N);
	yvals = _.map(xvals,function(x) { return evalCubic(x,a,b,c,d)});
	zipped = _.zip(xvals,yvals);
	var coords = new Array();
	_.map(zipped, function(p) {
		coords.push({x: p[0], y: p[1]});
		
	});
	return coords;
}

// y = genCubic(1,5,10,1,2,3,4);
// console.log(y);
function genPath(type,x0,x1,N,params) {
	var p = new paper.Path();
	switch (type) {
		case 'cubic':
			a = params[0];
			b = params[1];
			c = params[2];
			d = params[3];	 
			coords = genCubic(x0,x1,N,a,b,c,d);
			break;
	}
	return coords;		
}

function genOffsetPath(offset,mainPath) {
	p = new paper.Path();

}


////////// GENERATE POLYNOMIAL PATH ////////// 

// canvas params
width = 500;
height = 500;

// set up paper
paper.setup([width,height]);

// generate points along polynomial
coords = genPath('cubic',0,500,4,[0,0.0001,0,0]);

////////// APPLY GAUSSIAN BLUR TO PATH //////////

// adapted from paper.js smooth method
// line 5628 in paper-full.js
// takes in: x,y coords
// outputs: segments (point + handleIn + handleOut)

// adapted from catmullFitter.js
function catmullFitter2(coords,alpha) {

    if (alpha == 0 || alpha === undefined) {
      return false;
    } else {
      var p0, p1, p2, p3, d1, d2, d3;
      var p = new Array();
      var handleIn = new Array ();
      var handleOut = new Array();
      var segments = new Array();
      var d = 'M' + Math.round(coords[0].x) + ',' + Math.round(coords[0].y) + ' ';
      var length = coords.length;
      for (var i = 0; i < length - 1; i++) {
	        p0 = i == 0 ? coords[0] : coords[i - 1];
	        p1 = coords[i];
	        p2 = coords[i + 1];
	        p3 = i + 2 < length ? coords[i + 2] : p2;

	        d1 = dist(p0.x, p0.y, p1.x, p1.y);
	        d2 = dist(p1.x, p1.y, p2.x, p2.y);
	        d3 = dist(p2.x, p2.y, p3.x, p3.y);

	        var a = alpha,
				d1_a = Math.pow(d1, a),
				d1_2a = d1_a * d1_a,
				d2_a = Math.pow(d2, a),
				d2_2a = d2_a * d2_a;

			// set starting point
			p.push({
				x: coords[i].x,
				y: coords[i].y
			})

			// set handleIn
			var A = 2 * d2_2a + 3 * d2_a * d1_a + d1_2a,
				N = 3 * d2_a * (d2_a + d1_a);
			
			handleIn.push({
				x: (d2_2a * p0.x + A * p1.x - d1_2a * p2.x) / N - p1.x,
				y: (d2_2a * p0.y + A * p1.y - d1_2a * p2.y) / N - p1.y
			});

			// set handleOut
			var A = 2 * d1_2a + 3 * d1_a * d2_a + d2_2a,
				N = 3 * d1_a * (d1_a + d2_a);

			handleOut.push({
				x: (d1_2a * p2.x + A * p1.x - d2_2a * p0.x) / N - p1.x,
				y: (d1_2a * p2.y + A * p1.y - d2_2a * p0.y) / N - p1.y
			})
	  }

	}
	var segments = {
		points: p,
		handleIn: handleIn,
		handleOut: handleOut
	};
	return segments;

}

segments = catmullFitter2(coords, 0.5);
console.log(segments);

///// ///// ///// RENDERING STEP ///// ///// /////

// generate SVG string of catmull-rom-fitted spline
pathData = catmullFitter(coords,0.5);

// pass SVG string to paper Path obj
var path = new paper.Path(pathData);
path.strokeColor = 'black';

// save out SVG string as PNG

