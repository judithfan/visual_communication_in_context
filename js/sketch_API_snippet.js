        // define sketchpad
        var editor = Raphael.sketchpad("editor", {
              height: 500,
              width: 500,
              editing: true
          });


        // When the sketchpad changes, update the input field.
        editor.change(function() {
          $("#data").val(editor.json());
        });


        // experiment stuff happens... 

        // then submitResponse is called
        function submitResponse(){
            clickToggle(1); // this turns upload submission off
            outSubmitButton();

            var timestamp_sketchSubmit = new Date().getTime();
            console.log('Saving...');

            // TO CONVERT SVG TO PNG
            paper = editor.paper();
            var json = editor.json();
            json = paper.serialize.json(); // saves as json
            var svgString = paper.toSVG();
            a = document.createElement('a');
            a.type = 'image/svg+xml';
            blob = new Blob([svgString], {"type": "image/svg+xml"});
            a.href = (window.URL || webkitURL).createObjectURL(blob);
            svgUri = a.href;

            var canvas = document.querySelector("canvas"),
            context = canvas.getContext("2d");

            var image = new Image;
            image.onload = imageOnLoad;
            image.onerror=function(){console.log("Image failed to load")};
            image.src = a.href;

       function imageOnLoad(){
            context.drawImage(image,0,0,500,500);
            pngUrl = canvas.toDataURL();
            pngUrl = pngUrl.replace('data:image/png;base64,','');

            current_data = {imgData: pngUrl,
                           json: json,
                           colname:'graphcomm_explore_splines',
                           dbname:'splines',
                           trialNum: counter,
                           };

           $.ajax({
                   type: 'GET',
                   url: 'http://18.93.15.28:9919/saveimage',
                   dataType: 'jsonp',
                   traditional: true,
                   contentType: 'application/json; charset=utf-8',
                   data: current_data,
            success: function(msg) {console.log('image uploaded successfully');}

            });

