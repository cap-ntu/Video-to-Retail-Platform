import {
  PROD_INSERT_RECEIVE,
  PROD_INSERT_REQUEST,
  PROD_SEARCH_RECEIVE,
  PROD_SEARCH_REQUEST,
  PROD_BIDDING_REQUEST,
  PROD_BIDDING_RECEIVE,
} from '../../../constants/actionTypes';
import { asyncReducer } from '../utils';

const initState = {
  search: {
    state: 'INIT',
    time: null,
    reason: '',
    videos: {},
  },
  insert: {
    state: 'INIT',
    time: null,
    reason: '',
  },
  bidding: {
    state: 'INIT',
    time: null,
    reason: '',
    data: {},
  },
};

const product = (state = initState, action) => {
  let plugin = {};

  switch (action.type) {
    case PROD_SEARCH_REQUEST:
    case PROD_SEARCH_RECEIVE:
      return {
        ...state,
        search: asyncReducer(
          state.search,
          action,
          {
            request: PROD_SEARCH_REQUEST,
            receive: PROD_SEARCH_RECEIVE,
          },
          { videos: action.data },
        ),
      };
    case PROD_INSERT_REQUEST:
    case PROD_INSERT_RECEIVE:
      return {
        ...state,
        insert: asyncReducer(state.insert, action, {
          request: PROD_INSERT_REQUEST,
          receive: PROD_INSERT_RECEIVE,
        }),
      };
    case PROD_BIDDING_REQUEST:
    case PROD_BIDDING_RECEIVE:
      if (action.status === 'SUCCESS') {
        const { videos } = state.search;
        const { id, price } = action.data;
        // reformat the videos
        const clip = Object.values(videos)
          .reduce((x, y) => x.concat(y), [])
          .filter(x => x.id === id)[0];
        clip.price = price;
        plugin = {
          search: {
            ...state.search,
            videos: { ...videos },
          },
        };
      }
      return {
        ...state,
        ...plugin,
        bidding: asyncReducer(state.bidding, action, {
          request: PROD_BIDDING_REQUEST,
          receive: PROD_BIDDING_RECEIVE,
        }),
      };
    default:
      return { ...state };
  }
};

export default product;
