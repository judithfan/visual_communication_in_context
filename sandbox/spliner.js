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

// test paper pathmaker
paper.setup([100,100]);

// coords = genPath();
// console.log(coords);

// path.add([50,60]);
// path.add([30,80]);
// path.smooth(10);

// console.log(path._segments);


// test catmull fitter
coords = genPath('cubic',0,600,5,[0,0.001,0,0]);
// console.log(coords);
pathData = catmullFitter(coords,0.5);// console.log(coords);
var path = new paper.Path(pathData);

path.strokeColor = 'black';

// Scale the copy by 1000%, so we see something
path.scale(10);
console.log(pathData);
// naively just pass SVG string to path item
// Item = path.importSVG(svgString);

// console.log(path._segments);
