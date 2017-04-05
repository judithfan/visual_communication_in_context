// sketcher.js 
// differentiable spline renderer, built from adnn + paper
// jefan 4/4/17

var ad = require('adnn/ad')
var paper = require('paper')


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
    val = 1/(s*Math.sqrt(2*Math.PI)) * Math.exp(- Math.pow(m-t,2)/(2 * Math.pow(s,2)));
    return val;
}

// gS = genStops
// generate gradient stops for normal vector to main path
// t = distance from main path
// s = standard deviation of gaussian 
function gS(t,s) { 
    n = eG(0,0,s);
    c = 1 - eG(t,0,s)/n;
    var s = new Color(c,c,c);
    console.log(c);
    return s;
}

// 

paper.setup([100,100]);
var path = new paper.Path();
path.add([50,60]);
path.add([30,80]);
path.smooth(10);

console.log(path._segments);