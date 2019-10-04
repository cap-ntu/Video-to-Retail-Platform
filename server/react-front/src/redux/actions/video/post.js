import {
  VIDEO_UPLOAD_RECEIVE,
  VIDEO_UPLOAD_REQUEST,
} from '../../../constants/actionTypes';
import { xmlHttpRequest } from '../../../utils/httpRequest';
import { emptyFunction } from '../../../utils/utils';

const requestPost = () => ({
  type: VIDEO_UPLOAD_REQUEST,
});

const receivePost_Success = json => ({
  type: VIDEO_UPLOAD_RECEIVE,
  status: 'SUCCESS',
  data: json.data,
  time: json.time,
});

const receivePost_Failure = json => ({
  type: VIDEO_UPLOAD_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
});

export const VIDEO_upload = ({
  file,
  process,
  modelTypes,
  loadListener = emptyFunction,
  errorListener = emptyFunction,
  progressListener,
  abortListener,
}) => dispatch => {
  function modelMapper(models) {
    return Object.keys(models).filter(modelName => models[modelName]);
  }

  const formData = new FormData();
  formData.append('video_file', file, file.name);
  formData.append('name', file.name);
  if (process) formData.append('if_process', 'true');

  Object.values(modelTypes)
    .map(modelMapper)
    .reduce((x, y) => x.concat(y))
    .forEach(model => formData.append('models', model.toString()));

  const request = xmlHttpRequest(dispatch, 'POST', {
    url: '/restapi/videos/',
    request: requestPost,
    receiveSuccess: json => {
      loadListener();
      return receivePost_Success(json);
    },
    receiveFailure: json => {
      errorListener();
      return receivePost_Failure(json);
    },
    progressListener,
    formData,
  });

  return {
    abort: () => {
      request.abort();
      abortListener();
    },
  };
};
