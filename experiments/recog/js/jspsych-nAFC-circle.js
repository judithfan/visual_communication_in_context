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

(function($) {
  jsPsych["nAFC-circle"] = (function() {

    var plugin = {};

    plugin.create = function(params) {

      var trials = new Array(params.options.length);

      for (var i = 0; i < trials.length; i++) {
        trials[i] = {};
        trials[i].set_size = params.set_size;
        trials[i].num_trials = params.num_trials;
        trials[i].target = params.target[i];
        trials[i].sketch = params.sketch[i];
        trials[i].category = params.category_list,
        trials[i].distractor1 = params.distractor1,
        trials[i].distractor2 = params.distractor2,
        trials[i].distractor3 = params.distractor3,
        trials[i].context = params.context,
        trials[i].draw_duration = params.draw_duration,
        trials[i].num_strokes = params.num_strokes,
        trials[i].viewer_correct_in_context = params.viewer_correct_in_context,
        trials[i].viewer_response_in_context = params.viewer_response_in_context,
        trials[i].viewer_RT_in_context = params.viewer_RT_in_context,
        trials[i].gameID = params.gameID,
        trials[i].object_size = params.object_size || [100, 100];
        trials[i].sketch_size = params.sketch_size || [16, 16];
        trials[i].circle_diameter = params.circle_diameter || 250;
        trials[i].timing_max_search = (typeof params.timing_max_search === 'undefined') ? -1 : params.timing_max_search;
        trials[i].timing_sketch = (typeof params.timing_sketch === 'undefined') ? 500 : params.timing_sketch;
        trials[i].options = params.options || ['./object/dogs_08_pug_0035.png'];
      }

      return trials;
    };

    plugin.trial = function(display_element, trial) {

      trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

      // screen information
      var screenw = display_element.width();
      var screenh = display_element.height();
      var centerx = screenw / 2;
      var centery = screenh / 2;

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
          var start_time = Date.now();
          var end_time = Date.now();
          var rt = end_time - start_time;
          console.log('choice',choice);
          console.log('trial.target',trial.target);
          var correct = 0;
          if (choice == trial.target) {
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

      function end_trial(rt, correct, key_press) {

        // data saving
        var trial_data = {
          correct: correct,
          rt: rt,
          locations: JSON.stringify(display_locs),
          sketch: trial.sketch,
          target: trial.target
        };

        // this line merges together the trial_data object and the generic
        // data object (trial.data), and then stores them.
        jsPsych.data.write(trial_data);

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
})(jQuery);
