import {
  PROD_BIDDING_RECEIVE,
  PROD_BIDDING_REQUEST,
  PROD_INSERT_RECEIVE,
  PROD_INSERT_REQUEST,
  PROD_SEARCH_RECEIVE,
  PROD_SEARCH_REQUEST,
} from '../../../constants/actionTypes';
import { xmlHttpRequest } from '../../../utils/httpRequest';
import {
  decodeBiddingJson,
  decodeSceneSearchJson,
} from '../../../utils/jsonDecoder';

const requestSearch = () => ({
  type: PROD_SEARCH_REQUEST,
});

const receiveSearchSuccess = json => ({
  type: PROD_SEARCH_RECEIVE,
  status: 'SUCCESS',
  data: json.data,
  time: json.time,
});

const receiveSearchFailure = json => ({
  type: PROD_SEARCH_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
});

export const PRODsearch = ({ text, img, face, target }) => dispatch => {
  const formData = new FormData();

  if (text) formData.append('text', text);
  if (img) formData.append('img', img, img.name);
  if (face) formData.append('face', face);
  if (target && target.length) formData.append('target_videos', target);

  xmlHttpRequest(dispatch, 'POST', {
    url: '/restapi/scenesearch/',
    request: requestSearch,
    jsonDecoder: decodeSceneSearchJson,
    receiveSuccess: receiveSearchSuccess,
    receiveFailure: receiveSearchFailure,
    formData,
  });

  // dispatch(receiveSearchSuccess({ data: decodeSceneSearchJson(require('../../../asset/scenesearch.json').data) }));
};

const requestInsert = () => ({
  type: PROD_INSERT_REQUEST,
});

const receiveInsertSuccess = json => ({
  type: PROD_INSERT_RECEIVE,
  status: 'SUCCESS',
  data: json.data,
  time: json.time,
});

const receiveInsertFailure = json => ({
  type: PROD_INSERT_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
});

export const PRODinsert = id => dispatch => {
  const formData = new FormData();

  formData.append('scene_id', id);

  xmlHttpRequest(dispatch, 'POST', {
    url: '/restapi/productinsert/',
    request: requestInsert,
    receiveSuccess: receiveInsertSuccess,
    receiveFailure: receiveInsertFailure,
    formData,
  });
};

const requestBidding = () => ({
  type: PROD_BIDDING_REQUEST,
});

const receiveBiddingSuccess = json => ({
  type: PROD_BIDDING_RECEIVE,
  status: 'SUCCESS',
  data: json.data,
  time: json.time,
});

const receiveBiddingFailure = json => ({
  type: PROD_BIDDING_RECEIVE,
  status: 'FAILURE',
  reason: json.data,
});

export const PRODinitBiddingWebsocket = (message, dispatch) => {
  const data = JSON.parse(message.data);
  if (data.error) dispatch(receiveBiddingFailure({ reason: data.error }));
  switch (data.msg_type) {
    case 'connected':
      dispatch(requestBidding());
      break;
    case 'update':
      dispatch(receiveBiddingSuccess({ data: decodeBiddingJson(data) }));
      break;
    default:
      dispatch(receiveBiddingFailure({ reason: 'Unknown message type' }));
  }
};
