// Round float
function round(value, precision) {
  if (Number.isInteger(precision)) {
    var shift = Math.pow(10, precision);
    return Math.round(value * shift) / shift;
  } else {
    return Math.round(value);
  }
}

// Draw bounding box
function drawBox(box, class_name, confidence, ctx) {
  var ymin = box[0] * 365,
    xmin = box[1] * 640,
    ymax = box[2] * 365,
    xmax = box[3] * 640;
  var width = xmax - xmin;
  var height = ymax - ymin;
  ctx.rect(xmin, ymin, width, height);
  ctx.stroke();
  if(class_name) {
    ctx.fillText(class_name, xmin + 2, ymin + 10);
  }
  if(confidence) {
    ctx.fillText(confidence, xmin + 2, ymin + 20);
  }
}

window.onload = function() {
  // Video
  var video = document.getElementById("video");

  // Buttons
  var playButton = document.getElementById("play-pause");
  var muteButton = document.getElementById("mute");
  var fullScreenButton = document.getElementById("full-screen");
  var bboxToggleButton = document.getElementById("bbox-toggle");
  // Sliders
  var seekBar = document.getElementById("seek-bar");
  var volumeBar = document.getElementById("volume-bar");

  if(is_processed === true) {
      // Initiate scene index
      var cur_scene_idx = -1;
      var cur_scene_end = -1;

      // Instantiate canvas context
      var canvas = document.getElementById("overlay");
      var ctx = canvas.getContext("2d");
      ctx.font = "3x Arial";
      ctx.fillStyle = "blue";
      ctx.strokeStyle = "#FF0000";

      // Create waveform
      var wavesurfer = WaveSurfer.create({
        container: '#waveform',
        scrollParent: true
      });
      wavesurfer.load(audio_path);
      wavesurfer.setMute(true);

      // Add seek event listener for waveform
      wavesurfer.on("seek", function(cur_seek) {
        // Update time seek bar
        var time = video.duration * cur_seek;
        // Update the video time
        video.currentTime = time;
      });

  }
  // Event listener for the play/pause button
  playButton.addEventListener("click", function() {
    if (video.paused == true) {
      // Play the video
      video.play();
      // Play audio player
      wavesurfer.play();
      // Update the button text to 'Pause'
      playButton.innerHTML = "Pause";
    } else {
      // Pause the video
      video.pause();
      // Pause the audio
      wavesurfer.pause();
      // Update the button text to 'Play'
      playButton.innerHTML = "Play";
    }
  });

  // Event listener for the mute button
  muteButton.addEventListener("click", function() {
    if (video.muted == false) {
      // Mute the video
      video.muted = true;

      // Update the button text
      muteButton.innerHTML = "Unmute";
    } else {
      // Unmute the video
      video.muted = false;

      // Update the button text
      muteButton.innerHTML = "Mute";
    }
  });

  // Event listener for the full-screen button
  fullScreenButton.addEventListener("click", function() {
    if (video.requestFullscreen) {
      video.requestFullscreen();
    } else if (video.mozRequestFullScreen) {
      video.mozRequestFullScreen(); // Firefox
    } else if (video.webkitRequestFullscreen) {
      video.webkitRequestFullscreen(); // Chrome and Safari
    }
  });

  // Event listener for the bbox-toggle button
  bboxToggleButton.addEventListener("click", function() {
    if (bboxToggleButton.innerHTML === "Bbox-Off") {
      bboxToggleButton.innerHTML = "Bbox-On";
      // Clear bounding boxes
      if(is_processed) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.beginPath();
      }
    } else {
      bboxToggleButton.innerHTML = "Bbox-Off";
      if(is_processed) {
          // Display for object detection
          document.querySelectorAll("#object_table tbody tr").forEach(e => {
            var tds = e.getElementsByTagName("td");
            var box = [];
            for (var i = 0; i < 4; i++) {
              box[i] = tds[i + 4].innerHTML;
            }
            var class_name = tds[2].innerHTML;
            var confidence = tds[3].innerHTML;
            drawBox(box=box, class_name=class_name, confidence=confidence, ctx=ctx);
          });
          // Display for text detection
          document.querySelectorAll("#text_table tbody tr").forEach(e => {
            var tds = e.getElementsByTagName("td");
            var box = [];
            for (var i = 0; i < 4; i++) {
              box[i] = tds[i + 1].innerHTML;
            }
            drawBox(box=box, class_name=null, confidence=null, ctx=ctx);
          });
          // Display for face detection
          document.querySelectorAll("#face_table tbody tr").forEach(e => {
            var tds = e.getElementsByTagName("td");
            var box = [];
            for (var i = 0; i < 4; i++) {
              box[i] = tds[i + 1].innerHTML;
            }
            drawBox(box=box, class_name=null, confidence=null, ctx=ctx);
          });
      }
    }
  });

  // Event listener for the seek bar
  seekBar.addEventListener("change", function() {
    // Calculate the new time
    var fraction = seekBar.value / frame_cnt;
    var time = video.duration * (fraction);
    // Update the video time
    video.currentTime = time;
    // Update waveform
    wavesurfer.seekTo(fraction);
  });

  // Update the seek bar and prediction result visualization as time updates
  video.addEventListener("timeupdate", function() {
    // Calculate the slider value
    var value = Math.ceil(frame_cnt * video.currentTime / video.duration);
    // Update the slider value
    seekBar.value = value;
    // var fraction = value / frame_cnt;
    // wavesurfer.seekTo(fraction);
    // Draw bounding box
    if (is_processed === true && value <= frame_cnt) {
        // Get json for the current frame
        var json = json_set[parseInt(value)];
        // Clear bounding boxes from last frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        if(json["detection_boxes"]) {
            // Rendering the object table
            var tbody = document
              .getElementById("object_table")
              .getElementsByTagName("tbody")[0];
            var rowCount = tbody.rows.length;
            // Delete rows from last rendering
            while (rowCount--) tbody.deleteRow(rowCount - 1);
            // Insert rows in table
            for (var i = 0; i < json["detection_boxes"].length; i++) {
              var newRow = document.createElement("tr");
              // id
              var cell = document.createElement("td");
              cell.appendChild(document.createTextNode(i));
              newRow.appendChild(cell);
              // class id
              cell = document.createElement("td");
              cell.appendChild(document.createTextNode(json["detection_classes"][i]));
              newRow.appendChild(cell);
              // class name
              cell = document.createElement("td");
              cell.appendChild(
                document.createTextNode(json["detection_classes_names"][i])
              );
              newRow.appendChild(cell);
              // confidence
              cell = document.createElement("td");
              cell.appendChild(
                document.createTextNode(round(json["detection_scores"][i], 2))
              );
              newRow.appendChild(cell);
              // bounding box
              for (var j = 0; j < 4; j++) {
                cell = document.createElement("td");
                cell.appendChild(
                  document.createTextNode(round(json["detection_boxes"][i][j], 2))
                );
                newRow.appendChild(cell);
              }
              tbody.appendChild(newRow);
            }
            // Add event listener for each row
            document.querySelectorAll("#object_table tbody tr").forEach(e =>
              e.addEventListener("click", function() {
                var tds = this.getElementsByTagName("td");
                var box = [];
                for (var i = 0; i < 4; i++) {
                  box[i] = tds[i + 4].innerHTML;
                }
                var class_name = tds[2].innerHTML;
                var confidence = tds[3].innerHTML;
                drawBox(box=box, class_name=class_name, confidence=round(confidence, 2), ctx=ctx);
              })
            );

            // Draw bounding box only when bbox-toggle is in 'Off' state
            if (bboxToggleButton.innerHTML === "Bbox-Off") {
              var boxes = json["detection_boxes"];
              var classes = json["detection_classes_names"];
              var confidences = json["detection_scores"];
              for (var i = 0; i < boxes.length; i++) {
                drawBox(box=boxes[i], class_name=classes[i], confidence=round(confidences[i], 2), ctx=ctx);
              }
            }
        }
        if(json["scene"]) {
            // Rendering the scene table
            var tbody = document
              .getElementById("scene_table")
              .getElementsByTagName("tbody")[0];
            var rowCount = tbody.rows.length;
            // Delete rows from last rendering
            while (rowCount--) tbody.deleteRow(rowCount - 1);
            // Insert rows in table
            for (var i = 0; i < json["scene"].length; i++) {
              var newRow = document.createElement("tr");
              // id
              var cell = document.createElement("td");
              cell.appendChild(document.createTextNode(i));
              newRow.appendChild(cell);
              // scene name
              cell = document.createElement("td");
              cell.appendChild(document.createTextNode(json["scene"][i]));
              newRow.appendChild(cell);
              // confidence
              cell = document.createElement("td");
              cell.appendChild(document.createTextNode(json["score"][i]));
              newRow.appendChild(cell);
              tbody.appendChild(newRow);
            }
        }
        if(json["text_bboxes"]) {
            // Rendering the text table
            var tbody = document
              .getElementById("text_table")
              .getElementsByTagName("tbody")[0];
            var rowCount = tbody.rows.length;
            // Delete rows from last rendering
            while (rowCount--) tbody.deleteRow(rowCount - 1);
            // Insert rows in table
            for (var i = 0; i < json["text_bboxes"].length; i++) {
              var newRow = document.createElement("tr");
              // id
              var cell = document.createElement("td");
              cell.appendChild(document.createTextNode(i));
              newRow.appendChild(cell);
              // bounding box
              for (var j = 0; j < 4; j++) {
                cell = document.createElement("td");
                cell.appendChild(
                  document.createTextNode(round(json["text_bboxes"][i][j], 2))
                );
                newRow.appendChild(cell);
              }
              tbody.appendChild(newRow);
            }
            // Add event listener for each row
            document.querySelectorAll("#text_table tbody tr").forEach(e =>
              e.addEventListener("click", function() {
                var tds = this.getElementsByTagName("td");
                var box = [];
                for (var i = 0; i < 4; i++) {
                  box[i] = tds[i + 1].innerHTML;
                }
                drawBox(box=box, class_name=null, confidence=null, ctx=ctx);
              })
            );
            // Draw bounding box only when bbox-toggle is in 'Off' state
            if (bboxToggleButton.innerHTML === "Bbox-Off") {
              var boxes = json["text_bboxes"];
              for (var i = 0; i < boxes.length; i++) {
                drawBox(box=boxes[i], class_name=null, confidence=null, ctx=ctx);
              }
            }
        }
        if(json["face_bboxes"]) {
            // Rendering the text table
            var tbody = document
              .getElementById("face_table")
              .getElementsByTagName("tbody")[0];
            var rowCount = tbody.rows.length;
            // Delete rows from last rendering
            while (rowCount--) tbody.deleteRow(rowCount - 1);
            // Insert rows in table
            for (var i = 0; i < json["face_bboxes"].length; i++) {
              var newRow = document.createElement("tr");
              // id
              var cell = document.createElement("td");
              cell.appendChild(document.createTextNode(i));
              newRow.appendChild(cell);
              // person name
              cell = document.createElement("td");
              cell.appendChild(document.createTextNode(json["face_names"][i]));
              newRow.appendChild(cell);
              // bounding box
              for (var j = 0; j < 4; j++) {
                cell = document.createElement("td");
                cell.appendChild(
                  document.createTextNode(round(json["face_bboxes"][i][j], 2))
                );
                newRow.appendChild(cell);
              }
              tbody.appendChild(newRow);
            }
            // Add event listener for each row
            document.querySelectorAll("#face_table tbody tr").forEach(e =>
              e.addEventListener("click", function() {
                var tds = this.getElementsByTagName("td");
                var box = [];
                for (var i = 0; i < 4; i++) {
                  box[i] = tds[i + 1].innerHTML;
                }
                drawBox(box=box, class_name=json["face_names"][i], confidence=null, ctx=ctx);
              })
            );
            // Draw bounding box only when bbox-toggle is in 'Off' state
            if (bboxToggleButton.innerHTML === "Bbox-Off") {
              var boxes = json["face_bboxes"];
              for (var i = 0; i < boxes.length; i++) {
                drawBox(box=boxes[i], class_name=json["face_names"][i], confidence=null, ctx=ctx);
              }
            }
        }
        if(audio_enabled === true) {
            // Divide the index by frame rate
            var frame_idx = parseInt(video.currentTime * fr);
            if(frame_idx < audio_json_set.length) {
                cur_frame = audio_json_set[frame_idx];
                var ul = document.getElementById("audio_scene_prediction");
                // Clear list
                ul.innerHTML = "";
                for(var i = 0; i < cur_frame["labels"].length; i++) {
                  var li = document.createElement("li");
                  li.appendChild(document.createTextNode(cur_frame["labels"][i] + ": " + round(cur_frame["scores"][i], 4)));
                  ul.appendChild(li);
                }
            }
        }
        if(value > cur_scene_end) {
          cur_scene_idx += 1;
          // Get Statistics for the first scene
          var cur_scene = statistics[cur_scene_idx];
          cur_scene_end = cur_scene["end_frame"];
          // Sort statistics
          var sortable = [];
          for (var obj in cur_scene["cur_scene_statistics"]) {
            sortable.push([obj, cur_scene["cur_scene_statistics"][obj]]);
          }
          sortable.sort(function(a, b) {
            return b[1] - a[1];
          });
          // Display statistics
          var caption = document.getElementById("statistics_table").getElementsByTagName("caption")[0];
          caption.innerHTML = "SCENE " + cur_scene_idx;
          var tbody = document.getElementById("statistics_table").getElementsByTagName("tbody")[0];
          // Delete previous rows
          var rowCount = tbody.rows.length;
          // Delete rows from last rendering
          while (rowCount--) tbody.deleteRow(rowCount - 1);

          for (var i = 0; i < sortable.length; i++) {
            var newRow = document.createElement("tr");
            var cell = document.createElement("td");
            cell.appendChild(document.createTextNode(sortable[i][0]));
            newRow.appendChild(cell);
            cell = document.createElement("td");
            cell.appendChild(document.createTextNode(sortable[i][1]));
            newRow.appendChild(cell);
            tbody.appendChild(newRow);
          }
        }
    }
  });

  // Pause the video when the seek handle is being dragged
  seekBar.addEventListener("mousedown", function() {
    video.pause();
  });

  // Play the video when the seek handle is dropped
  seekBar.addEventListener("mouseup", function() {
    video.play();
  });

  // Event listener for the volume bar
  volumeBar.addEventListener("change", function() {
    // Update the video volume
    video.volume = volumeBar.value;
  });
};
