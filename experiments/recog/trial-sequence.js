
// 0. Load dependencies
paper.install(window);
socket = io.connect();



// screen information
var screenw = display_element.width();
var screenh = display_element.height();
var centerx = screenw / 2;
var centery = screenh / 2;

// circle params
var diam = trial.circle_diameter; // pixels
var radi = diam / 2;
var paper_size = diam + trial.target_size[0];

// stimuli width, height
var stimh = trial.target_size[0];
var stimw = trial.target_size[1];
var hstimh = stimh / 2;
var hstimw = stimw / 2;

// fixation location
var fix_loc = [Math.floor(paper_size / 2 - trial.fixation_size[0] / 2), Math.floor(paper_size / 2 - trial.fixation_size[1] / 2)];

// possible stimulus locations on the circle
var display_locs = [];
var possible_display_locs = trial.set_size;
var random_offset = Math.floor(Math.random() * 360);
for (var i = 0; i < possible_display_locs; i++) {
display_locs.push([
  Math.floor(paper_size / 2 + (cosd(random_offset + (i * (360 / possible_display_locs))) * radi) - hstimw),
  Math.floor(paper_size / 2 - (sind(random_offset + (i * (360 / possible_display_locs))) * radi) - hstimh)
]);
}
