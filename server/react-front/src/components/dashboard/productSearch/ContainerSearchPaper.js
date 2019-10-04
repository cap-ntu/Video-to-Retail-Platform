import { connect } from 'react-redux';
import { PRODsearch } from '../../../redux/actions/product/post';
import { VIDEOgetList } from '../../../redux/actions/video';
import SearchPaper from './search/SearchPaper';

const mapStateToProps = state => ({
  videos: state.video.videoList.videos,
});

const mapDispatchToProps = dispatch => ({
  handleSearch: props => dispatch(PRODsearch(props)),
  handleGetVideoList: successCallback =>
    dispatch(VIDEOgetList(successCallback)),
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(SearchPaper);
