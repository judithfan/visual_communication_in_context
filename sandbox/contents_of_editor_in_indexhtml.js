var path;
var path2;
var p;

// The mouse has to drag at least N pt
// before the next drag event is fired:
tool.minDistance = 1;
paper.install(window);

function eG(t,m,s) { // eG = evaluateGaussian
    val = 1/(s*Math.sqrt(2*Math.PI)) * Math.exp(- Math.pow(m-t,2)/(2 * Math.pow(s,2)));
    return val;
}

function gS(t,s) { // gS = genStops
    n = eG(0,0,s);
    c = 1 - eG(t,0,s)/n;
    var s = new Color(c,c,c);
    // console.log(c);
    return s;
}

function onMouseDown(event) {
    if (path) {
        path.selected = false;
    };
    path = new Path();
    path2 = new Path();
    _p = new Path();
    // path2.strokeColor = 'red';
    path.strokeColor = 'black';
    path.strokeWidth = 5;
    path2.strokeWidth = 5;
    path.fullySelected = true;
}

function onMouseDrag(event) {
    path.add(event.point); 
    
    var p2 = new Path();
    p2.strokeWidth = 10;
    p2.strokeColor = 'black';
    p2.opacity = 0.5;
    var vector = event.delta;

    // rotate the vector by 90 degrees:
    vector.angle += 90;

    // change its length to 5 pt:
    vector.length = 30;
    
    p2.add(event.middlePoint + vector);
    p2.add(event.middlePoint - vector);  
    
    p2.strokeColor = {
        gradient: {
            stops: [[gS(-4,1), 0], [gS(-3,1), 0.125], [gS(-2,1), 0.25],[gS(-1,1), 0.375],
                    [gS(0,1), 0.5], [gS(1,1), 0.625], [gS(2,1), 0.75],[gS(3,1), 0.875],[gS(4,1), 1]],
            radial: false
        },
        origin: event.middlePoint + vector,
        destination: event.middlePoint - vector
    };  
    
    
}

function onMouseUp(event) {
    path.selected = false;
    path.smooth(10);
    norm = path.getNormalAt(path.length)*30;
    // norm2 = norm.rotate(180); // get the other normal    
    finalPoint = path._segments.slice(-1)[0];
    // console.log(norm);
    // console.log(finalPoint['point']+norm)
    path2.add(finalPoint['point']-norm);
    path2.add(finalPoint['point']);
    path2.add(finalPoint['point']+norm);
    // console.log(path.exportJSON());
    
    svgString = paper.project.exportJSON({asString:true});
    var serializer = new XMLSerializer();
    var svg = paper.project.exportSVG();
    var svg_string = serializer.serializeToString(svg);  
    // console.log(svg_string);
    
    var blob = new Blob([svg_string], {"type": "image/svg+xml"});          
    // console.log(blob);    
    
    a = document.createElement('a');
    a.type = 'image/svg+xml';
    a.href = window.URL.createObjectURL(blob);
    
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");

    var img = new Image();
    img.onload = imageOnLoad; 
    img.onerror=function(){console.log("Image failed to load")};    
    img.src = a.href;
    
    
    function imageOnLoad() {
        ctx.drawImage(img, 0, 0);
        var myImageData = ctx.getImageData(0,0,100,100);
        pngUrl = canvas.toDataURL();
        pngUrl = pngUrl.replace('data:image/png;base64,','');
        var sum = _.reduce(myImageData.data, function(memo, num){ return memo + num; }, 0);
        // console.log(sum);
        // console.log(_.unique(myImageData.data));
        
    };
    

    path2.strokeColor = {
        gradient: {
            stops: [[gS(-4,1), 0], [gS(-3,1), 0.125], [gS(-2,1), 0.25],[gS(-1,1), 0.375],
                    [gS(0,1), 0.5], [gS(1,1), 0.625], [gS(2,1), 0.75],[gS(3,1), 0.875],[gS(4,1), 1]],
            radial: false
        },
        origin: finalPoint['point']-norm,
        destination: finalPoint['point']+norm
    };

    }
