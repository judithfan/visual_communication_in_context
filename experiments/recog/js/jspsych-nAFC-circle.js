/**
 *
 * jspsych-nAFC-circle
 * Judy Fan
 *
 * displays a target image at center surrounded by several unique images
 * positioned equidistant from the target.
 * participant's goal is to click on the surround image that best matches the target.
 *
 *
 * requires Snap.svg library (snapsvg.io)
 *
 * documentation: docs.jspsych.org || TBD
 *
 **/

jsPsych.plugins["nAFC-circle"] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'nAFC-circle',
    parameters: {      
    }
  }

  plugin.trial = function(display_element, trial) {

    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

    // screen information
    var screenw = display_element.width();
    var screenh = display_element.height();
    var centerx = screenw / 2;
    var centery = screenh / 2;

    // initialize start_time timestamp
    var start_time = Date.now();

    // circle params
    var diam = trial.circle_diameter; // pixels
    var shrinkage = 0.94; // amount by which to shrink the radius of the circle to not spill off paper
    var radi = diam / 2 * shrinkage;
    var paper_size = diam + trial.object_size[0];

    // stimuli width, height
    var stimh = trial.object_size[0];
    var stimw = trial.object_size[1];
    var hstimh = stimh / 2; 
    var hstimw = stimw / 2;

    // sketch location
    var fix_loc = [Math.floor(paper_size / 2 - trial.sketch_size[0] / 2), Math.floor(paper_size / 2 - trial.sketch_size[1] / 2)];

    // possible stimulus locations on the circle
    var display_locs = [];
    var possible_display_locs = trial.set_size;
    // var random_offset = Math.floor(Math.random() * 360);
    var random_offset = 0;
    for (var i = 0; i < possible_display_locs; i++) {
      display_locs.push([
        Math.floor(paper_size / 2 + (cosd(random_offset + (i * (360 / possible_display_locs))) * radi) - hstimw),
        Math.floor(paper_size / 2 - (sind(random_offset + (i * (360 / possible_display_locs))) * radi) - hstimh)
      ]);
    }

    // get target to draw on
    display_element.append($('<svg id="jspsych-nAFC-circle-svg" width=' + paper_size + ' height=' + paper_size + '></svg>'));
    var paper = Snap('#jspsych-nAFC-circle-svg');

    show_object_array();

    function show_sketch() {
      // show sketch
      var sketch = paper.image(trial.sketch, fix_loc[0], fix_loc[1], trial.sketch_size[0], trial.sketch_size[1]);
      var start_time = Date.now();
    }

    function show_object_array() {
      var object_array_images = [];
      img = new Array;
      for (var i = 0; i < display_locs.length; i++) {
        var img = paper.image(trial.options[i], display_locs[i][0], display_locs[i][1], trial.object_size[0], trial.object_size[1]);                
        object_array_images.push(img);
      }
      var trial_over = false;

      // group object images and add hover animation
      images = paper.g(paper.selectAll('image'));
      images.selectAll('image').forEach( function( el, index ) {
         el.hover( function() { el.animate({ transform: 's2,2' }, 100, mina.easein); },
                   function() { el.animate({ transform: 's1,1' }, 100 , mina.easein); }
          )
      } );

      // add click listener to the objects
      var a = document.getElementsByTagName('g')[0];
      imgs = a.children;

      for (var i = 0; i < display_locs.length; i++) {
        imgs[i].addEventListener('click', function (e) {
          var choice = e.currentTarget.getAttribute('href'); // don't use dataset for jsdom compatibility
          after_response(choice);
         })
      }

      var after_response = function(choice) {
        trial_over = true;
        // measure rt
        var end_time = Date.now();
        var rt = end_time - start_time;                    
        bare_choice = choice.split('/')[2].split('.')[0];
        console.log('choice',bare_choice);
        console.log('target',trial.target);         
        var correct = 0;
        if (bare_choice == trial.target) {
          correct = 1;
        }
        clear_display();
        end_trial(rt, correct, choice); 
      }

      function clear_display() {
        paper.clear();
      }

      // wait
      setTimeout(function() {
        // after wait is over
        show_sketch();
      }, trial.timing_sketch);        

    }

    function end_trial(rt, correct, choice) {

      // data saving
      var trial_data = {
        rt: rt,
        correct: correct,          
        choice: choice,
        locations: JSON.stringify(display_locs),
        sketch: trial.sketch,
        target: trial.target,
        category: trial.category,
        distractor1: trial.distractor1,
        distractor2: trial.distractor2,
        distractor3: trial.distractor3,
        context: trial.context,
        draw_duration: trial.draw_duration,
        num_strokes: trial.num_strokes,
        viewer_correct_in_context: trial.viewer_correct_in_context,
        viewer_response_in_context: trial.viewer_response_in_context,
        viewer_RT_in_context: trial.viewer_RT_in_context,
        gameID: trial.gameID,
        object_size: trial.object_size,
        sketch_size: trial.sketch_size,
        circle_diameter: trial.circle_diameter
      };

      console.log(trial_data);

      // this line merges together the trial_data object and the generic
      // data object (trial.data), and then stores them.
      jsPsych.data.write(trial_data);

      // this is where we want to write to the database
      

      // go to next trial
      jsPsych.finishTrial();
    }
  };

  // helper function for determining stimulus locations

  function cosd(num) {
    return Math.cos(num / 180 * Math.PI);
  }

  function sind(num) {
    return Math.sin(num / 180 * Math.PI);
  }

  return plugin;
})();
