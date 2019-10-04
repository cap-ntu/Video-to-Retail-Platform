/* eslint-disable no-unused-vars */
const type = ['OBJECT', 'TEXT', 'SCENE', 'FACE'];

function classifyResultByType(result) {
  const resultCopy = {};
  type.forEach(key => {
    resultCopy[key] = [];
  });
  result.forEach(model => {
    const typeCopy = model.type;
    resultCopy[typeCopy].push({ ...model });
    delete resultCopy[typeCopy].type;
  });

  return resultCopy;
}

export function decodeVideoListJson(data) {
  return data.map(video => ({
    id: video.id.toString(),
    name: video.name,
    cover: video.cover,
    processed: video.processed,
    isProcessing: video.being_processed,
  }));
}

function decodeProductJson(data = []) {
  return data
    .sort((x, y) => x.start_time < y.start_time)
    .map(product => ({
      poster: product.product_path,
      start: product.start_time / 1000,
      end: product.end_time / 1000,
    }));
}

function decodeVisual(data = []) {
  const initValue = {
    text_bboxes: [],
    scene: [],
    score: [],
    num_detections: 0,
    detection_boxes: [],
    detection_scores: [],
    detection_classes_names: [],
    face_bboxes: [],
    face_names: [],
  };

  const frameKeyToModelType = {
    detection_boxes: 'object',
    face_bboxes: 'face',
    text_bboxes: 'text',
    scene: 'scene',
  };

  const modelTypes = [];

  function transpose(visual) {
    const retVal = {};
    [...Array(3).keys()].forEach(i => {
      retVal[modelTypes[i]] = visual.map(row => row[i]);
    });
    return retVal;
  }

  function decodeBox(box) {
    return { y1: box[0], x1: box[1], y2: box[2], x2: box[3] };
  }

  // noinspection JSUnusedLocalSymbols
  function decodeObject(frame) {
    return [...Array(frame.num_detections).keys()].map(index => ({
      ...decodeBox(frame.detection_boxes[index]),
      score: frame.detection_scores[index],
      name: frame.detection_classes_names[index],
    }));
  }

  // noinspection JSUnusedLocalSymbols
  function decodeFace(frame) {
    return frame.face_bboxes.map((box, index) => ({
      ...decodeBox(box),
      name: frame.face_names[index].replace('_', ' '),
    }));
  }

  // noinspection JSUnusedLocalSymbols
  function decodeText(frame) {
    return (
      frame.text_bboxes.map(box => ({
        ...decodeBox(box),
        name: 'text',
      })) || []
    );
  }

  // noinspection JSUnusedLocalSymbols
  function decodeScene(frame) {
    return (
      frame.scene.map((scene, index) => ({
        name: scene.replace('_', ' '),
        score: frame.score[index],
      })) || []
    );
  }

  const frame = data[0];
  if (frame) {
    for (const [k, v] of Object.entries(frameKeyToModelType))
      if (frame[k]) modelTypes.push(v);
  }

  return transpose(
    data.map(_frame => {
      Object.keys(initValue).forEach(key => {
        if (!_frame[key]) _frame[key] = initValue[key];
      });

      return modelTypes.map(_type =>
        // eslint-disable-next-line no-eval
        eval(`decode${_type.charAt(0).toUpperCase() + _type.slice(1)}`)(_frame),
      );
    }),
  );
}

function decodeAudio(data = []) {
  return data.map(frame =>
    frame.labels.map((key, index) => ({
      name: key,
      score: frame.scores[index],
    })),
  );
}

function decodeStatistics(data = []) {
  return data.map(scene => {
    const result = scene.cur_scene_statistics;

    return {
      start: scene.start_frame,
      end: scene.end_frame,
      results: Object.keys(result).map(key => {
        return {
          name: key,
          appearance: result[key],
        };
      }),
    };
  });
}

function decodeResultJson(data) {
  return {
    visual: decodeVisual(data.visual),
    audio: decodeAudio(data.audio),
    statistics: decodeStatistics(data.statistics),
  };
}

export function decodeSingleVideoJson(data) {
  return {
    url: data.url,
    audio: data.audio_url,
    pic: data.pic,
    processed: data.processed,
    isProcessing: data.being_processed,
    totalFrame: data.frame_cnt,
    models: data.models,
    result: decodeResultJson(data.result),
    products: decodeProductJson(data.products),
  };
}

export function decodeUserListJson(data) {
  data.sort((x, y) => x.username.localeCompare(y.username));
  return data.map(user => ({
    id: user.id,
    username: user.username,
    name: [user.first_name, user.last_name]
      .filter(i => i !== undefined)
      .join(' '),
    firstName: user.first_name,
    lastName: user.last_name,
    email: user.email,
    groups: user.groups,
    accountType: user.is_superuser ? 'Administrator' : 'User',
    domain: user.is_staff ? 'Staff' : 'Public',
    status: user.is_active ? 'Activated' : 'Not Activated',
  }));
}

export function decodeModelListJson(data) {
  return classifyResultByType(data);
}

export function decodeSingleModelJson(data) {
  return {
    id: data.id,
    name: data.name,
    type: data.type,
  };
}

export function decodeSceneSearchJson(data) {
  const result = {};

  data.forEach(video => {
    result[video[0].TV_NAME] = video.map(clip => ({
      id: clip.SCENE_ID,
      cover: clip.IMAGE,
      start: clip.START_TIME / 1000,
      end: clip.END_TIME / 1000,
      price: clip.HIGHEST_PRICE,
      name: clip.TV_NAME,
    }));
  });

  return result;
}

export function decodeBiddingJson(data) {
  return {
    id: data.scene_id,
    price: data.price,
  };
}
