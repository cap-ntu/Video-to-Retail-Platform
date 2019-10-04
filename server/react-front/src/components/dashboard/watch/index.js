import { connect } from 'react-redux';
import WatchApp from './WatchApp';
import { VIDEOgetSingle } from '../../../redux/actions/video/get';
import { VIDEO_clear } from '../../../redux/actions/video';

const mapStateToProps = state => ({
  video: state.video.singleVideo.video,
});

const mapDispatchToProps = dispatch => ({
  fetchVideoInfo: id => dispatch(VIDEOgetSingle(id)),
  clearVideoInfo: () => dispatch(VIDEO_clear()),
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(WatchApp);
