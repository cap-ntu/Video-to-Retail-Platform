import {
  VIDEO_CLEAR,
  VIDEO_DELETE_RECEIVE,
  VIDEO_DELETE_REQUEST,
  VIDEO_GET_LIST_RECEIVE,
  VIDEO_GET_LIST_REQUEST,
  VIDEO_GET_SINGLE_RECEIVE,
  VIDEO_GET_SINGLE_REQUEST,
  VIDEO_UPDATE_RECEIVE,
  VIDEO_UPDATE_REQUEST,
  VIDEO_UPLOAD_RECEIVE,
  VIDEO_UPLOAD_REQUEST,
} from '../../../constants/actionTypes';
import { asyncReducer } from '../utils';

const initState = {
  videoList: {
    state: 'INIT',
    time: null,
    reason: '',
    videos: [],
  },
  singleVideo: {
    state: 'INIT',
    time: null,
    video: {},
  },
  postVideo: {
    state: 'INIT',
    reason: '',
    time: null,
  },
  updateVideo: {
    state: 'INIT',
    reason: '',
    time: null,
  },
  deleteVideo: {
    state: 'INIT',
    reason: '',
    time: null,
  },
};

export default function video(state = initState, action) {
  switch (action.type) {
    case VIDEO_GET_LIST_REQUEST:
    case VIDEO_GET_LIST_RECEIVE:
      return {
        ...state,
        videoList: asyncReducer(
          state.videoList,
          action,
          {
            request: VIDEO_GET_LIST_REQUEST,
            receive: VIDEO_GET_LIST_RECEIVE,
          },
          {
            videos: action.videos,
          },
        ),
      };
    case VIDEO_GET_SINGLE_REQUEST:
    case VIDEO_GET_SINGLE_RECEIVE:
      return {
        ...state,
        singleVideo: asyncReducer(
          state.singleVideo,
          action,
          {
            request: VIDEO_GET_SINGLE_REQUEST,
            receive: VIDEO_GET_SINGLE_RECEIVE,
          },
          { video: action.video },
        ),
      };
    case VIDEO_UPLOAD_REQUEST:
    case VIDEO_UPLOAD_RECEIVE:
      return {
        ...state,
        postVideo: asyncReducer(state.postVideo, action, {
          request: VIDEO_UPLOAD_REQUEST,
          receive: VIDEO_UPLOAD_RECEIVE,
        }),
      };
    case VIDEO_UPDATE_REQUEST:
    case VIDEO_UPDATE_RECEIVE:
      return {
        ...state,
        updateVideo: asyncReducer(state.updateVideo, action, {
          request: VIDEO_UPDATE_REQUEST,
          receive: VIDEO_UPDATE_RECEIVE,
        }),
      };
    case VIDEO_DELETE_REQUEST:
    case VIDEO_DELETE_RECEIVE:
      return {
        ...state,
        deleteVideo: asyncReducer(state.deleteVideo, action, {
          request: VIDEO_DELETE_REQUEST,
          receive: VIDEO_DELETE_RECEIVE,
        }),
      };
    case VIDEO_CLEAR:
      return {
        ...state,
        singleVideo: {
          state: 'INIT',
          time: null,
          video: {},
        },
      };
    default:
      return { ...state };
  }
}
