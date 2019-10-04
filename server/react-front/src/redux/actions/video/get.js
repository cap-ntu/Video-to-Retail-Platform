import {
  VIDEO_GET_LIST_RECEIVE,
  VIDEO_GET_LIST_REQUEST,
  VIDEO_GET_SINGLE_RECEIVE,
  VIDEO_GET_SINGLE_REQUEST,
} from '../../../constants/actionTypes';
import { xmlHttpRequest } from '../../../utils/httpRequest';
import {
  decodeSingleVideoJson,
  decodeVideoListJson,
} from '../../../utils/jsonDecoder';
import { emptyFunction } from '../../../utils/utils';

const requestGetList = () => ({
  type: VIDEO_GET_LIST_REQUEST,
});

const receiveGetListSuccess = json => ({
  type: VIDEO_GET_LIST_RECEIVE,
  status: 'SUCCESS',
  videos: json.data,
  time: json.time,
});

const receiveGetListFailure = json => ({
  type: VIDEO_GET_LIST_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
});

export const VIDEOgetList = (successCallback = emptyFunction) => dispatch => {
  xmlHttpRequest(dispatch, 'GET', {
    url: '/restapi/videos/',
    jsonDecoder: decodeVideoListJson,
    request: requestGetList,
    receiveSuccess: json => {
      successCallback();
      return receiveGetListSuccess(json);
    },
    receiveFailure: receiveGetListFailure,
  });

  // dispatch(
  //   receiveGetListSuccess({
  //     data: decodeVideoListJson(require('../../../asset/videolist.json').data),
  //   }),
  // );
};

const requestGetSingle = () => ({
  type: VIDEO_GET_SINGLE_REQUEST,
});

const receiveGetSingleSuccess = json => ({
  type: VIDEO_GET_SINGLE_RECEIVE,
  status: 'SUCCESS',
  video: json.data,
  update: json.update,
});

const receiveGetSingleFailure = json => ({
  type: VIDEO_GET_SINGLE_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
  update: json.update,
});

export const VIDEOgetSingle = id => dispatch => {
  // const data = require("../../../asset/video_amine.json");

  xmlHttpRequest(dispatch, 'GET', {
    url: `/restapi/videos/${id}/`,
    jsonDecoder: decodeSingleVideoJson,
    request: requestGetSingle,
    receiveSuccess: receiveGetSingleSuccess,
    receiveFailure: receiveGetSingleFailure,
  });

  // dispatch(receiveGetSingle_Success({data: decodeSingleVideoJson(data.data)}))
};
