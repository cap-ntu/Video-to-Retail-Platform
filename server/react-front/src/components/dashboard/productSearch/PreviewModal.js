import React, { Component } from 'react';
import Modal from '@material-ui/core/es/Modal/Modal';
import * as PropTypes from 'prop-types';
import Button from '@material-ui/core/es/Button/Button';
import Player from '../common/player/Player';

const styles = {};

class PreviewModal extends Component {
  state = {
    time: 0,
  };

  componentWillMount() {
    this.props.fetchVideoInfo();
  }

  render() {
    const { classes, open, video, onClose, insertAd } = this.props;
    return (
      <Modal
        aria-labelledby='simple-modal-title'
        aria-describedby='simple-modal-description'
        open={open}
        onClose={onClose}
      >
        <Player
          boxOn={false}
          video={video}
          handleUpdateTime={time => this.setState({ time })}
        />
        />
        <Button onClick={insertAd}>Insert</Button>
      </Modal>
    );
  }
}

PreviewModal.propTypes = {
  classes: PropTypes.object.isRequired,
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  video: PropTypes.shape({
    url: PropTypes.string,
    poster: PropTypes.string,
    description: PropTypes.string,
  }).isRequired,
  fetchVideoInfo: PropTypes.func.isRequired,
  insertAd: PropTypes.func.isRequired,
};

export default withStyles(styles)(PreviewModal);
