import { connect } from 'react-redux';
import InsertionPointPaper from './InsertionPointPaper';
import {
  PRODinitBiddingWebsocket,
  PRODinsert,
} from '../../../redux/actions/product/post';

const mapStateToProps = state => ({
  videos: state.product.search.videos,
});

const mapDispatchToProps = dispatch => ({
  handleInsert: id => dispatch(PRODinsert(id)),
  websocketOnMessage: message => PRODinitBiddingWebsocket(message, dispatch),
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(InsertionPointPaper);
