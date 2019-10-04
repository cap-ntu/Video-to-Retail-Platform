import React from 'react';
import * as PropTypes from 'prop-types';
import Paper from '@material-ui/core/Paper';
import { FilePond } from 'react-filepond';
import Grid from '@material-ui/core/Grid';
import withStyles from '@material-ui/core/styles/withStyles';
import Typography from '@material-ui/core/Typography';
import Popper from '@material-ui/core/Popper';
import Grow from '@material-ui/core/Grow';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import FormLabel from '@material-ui/core/FormLabel';
import FormGroup from '@material-ui/core/FormGroup';
import FormHelperText from '@material-ui/core/FormHelperText';
import Link from '@material-ui/core/Link';
import Form from '../../../common/Form';

const styles = theme => ({
  root: {
    backgroundColor: 'transparent',
    width: '90%',
    margin: 'auto',
    padding: theme.spacing.unit,
  },
  card: {
    marginBottom: 2 * theme.spacing.unit,
    position: 'relative',
    '&:last-child': {
      marginBottom: 0,
    },
  },
  finish: {
    position: 'absolute',
    right: 0,
  },
  iconButton: {
    padding: 10,
  },
  tools: {
    marginTop: theme.spacing.unit,
  },
  filter: {
    opacity: 0.7,
    cursor: 'pointer',
    transition: [['opacity', '.2s']],
    userSelect: 'none',
    '&:hover': {
      opacity: 1,
    },
    '&:active *': {
      color: theme.palette.primary.main,
    },
  },
  menu: {
    width: 420,
    padding: theme.spacing.unit * 2,
    outline: 'none',
    borderRadius: theme.shape.buttonBorderRadius,
  },
  videoListRoot: {
    width: '100%',
    padding: theme.spacing.unit,
    maxHeight: 800,
    overflowY: 'scroll',
  },
});

const nameToState = {
  'Product Image': 'image',
  'Face Image': 'face',
  'Search From': 'video',
};

class SearchPaper extends Form {
  state = {
    anchorEl: null,
    mount: null,
    textContent: '',
    imageContent: [],
    faceContent: [],
    videoContent: [],
  };

  imageRef = React.createRef();

  faceRef = React.createRef();

  popupComponentMapper = key => {
    const { classes, videos } = this.props;
    const { imageContent, faceContent, videoContent } = this.state;

    switch (key) {
      case 'image':
        return (
          <FileUploader
            innerRef={this.imageRef}
            fileList={imageContent}
            updateList={this.handleFileChange('imageContent')}
          />
        );
      case 'face':
        return (
          <FileUploader
            innerRef={this.faceRef}
            fileList={faceContent}
            updateList={this.handleFileChange('faceContent')}
          />
        );
      case 'video':
        return (
          <VideoList
            classes={{ root: classes.videoListRoot }}
            videoContent={videoContent}
            videos={videos}
            updateSelection={this.handleVideoContentChange}
          />
        );
      default:
        return null;
    }
  };

  componentWillMount() {
    const { handleGetVideoList } = this.props;
    handleGetVideoList();
  }

  handleClose = e => {
    const { anchorEl } = this.state;

    if (anchorEl.contains(e.target)) {
      return;
    }

    this.setState({ anchorEl: null });
  };

  handleTextChange = (e, callback) => {
    this.setState({ textContent: e.target.value }, callback);
  };

  handleFileChange = name => (fileList, callback) => {
    this.setState({ [name]: fileList }, callback);
  };

  handleVideoContentChange = id => e => {
    const { checked } = e.target;
    this.setState(state => {
      const { videoContent: _v } = state;
      const videoContent = _v.slice();
      if (checked) videoContent.push(id);
      else {
        const index = videoContent.indexOf(id);
        if (index > -1) {
          videoContent.splice(index, 1);
        }
      }
      return { videoContent };
    });
  };

  handleSubmit = this.handleSubmitWrapper(() => {
    const { textContent, imageContent, videoContent } = this.state;
    this.props.handleSearch({
      text: textContent,
      img: imageContent.length ? imageContent[0] : null,
      target: videoContent,
    });
  });

  render() {
    const { classes, theme } = this.props;
    const { anchorEl, mount, textContent } = this.state;
    const open = Boolean(anchorEl);

    return (
      <Paper className={classes.root} elevation={0}>
        <Grid className={classes.tools} spacing={16} container>
          {Object.keys(nameToState).map(item => (
            <Grid
              className={classes.filter}
              key={item}
              onClick={e => {
                const { currentTarget } = e;
                this.setState(state => {
                  if (!state.anchorEl)
                    return {
                      mount: nameToState[item],
                      anchorEl: currentTarget,
                    };
                  return { anchorEl: null };
                });
              }}
              item
            >
              <Typography>
                {this.state[`${nameToState[item]}Content`].length ? '● ' : ''}
                {`${item} ▼`}
              </Typography>
            </Grid>
          ))}
        </Grid>

        <Popper
          open={open}
          anchorEl={anchorEl}
          onClose={this.handleClose}
          style={{ zIndex: theme.zIndex.drawer + 1 }}
          disablePortal
          transition
        >
          {({ TransitionProps }) => (
            <Grow
              {...TransitionProps}
              id='menu-list-grow'
              style={{
                transformOrigin: 'center top',
              }}
            >
              <Paper className={classes.menu}>
                <ClickAwayListener onClickAway={this.handleClose}>
                  {this.popupComponentMapper(mount)}
                </ClickAwayListener>
              </Paper>
            </Grow>
          )}
        </Popper>
      </Paper>
    );
  }
}

const FileUploader = ({ innerRef, fileList, updateList }) => (
  <FilePond
    ref={innerRef}
    oninit={() => innerRef.current.addFiles(fileList)}
    maxFiles={1}
    files={fileList}
    onupdatefiles={(items, callback) =>
      updateList(items.map(item => item.file), callback)
    }
    acceptedFileTypes={['image/*']}
  />
);

export const VideoList = ({
  classes,
  videos,
  videoContent,
  updateSelection,
}) => {
  const error = videoContent.length === 0;

  if (videos.length)
    return (
      <React.Fragment>
        <FormControl
          className={classes.root}
          component='fieldset'
          error={error}
          required
        >
          <FormLabel component='legend'>Video list</FormLabel>
          <FormGroup>
            {videos.map(v => (
              <FormControlLabel
                key={v.id}
                control={
                  // eslint-disable-next-line react/jsx-wrap-multilines
                  <Checkbox
                    checked={videoContent.includes(v.id)}
                    onChange={updateSelection(v.id)}
                    value='checked'
                  />
                }
                label={v.name}
              />
            ))}
          </FormGroup>
          <FormHelperText component='legend'>
            You have to select at least one video
          </FormHelperText>
        </FormControl>
      </React.Fragment>
    );
  // noinspection HtmlUnknownTarget
  return (
    <Typography align='center'>
      No video are available.
      <br />
      <Link href='/dashboard/video/upload'>Upload a new one?</Link>
    </Typography>
  );
};

FileUploader.propTypes = {
  innerRef: PropTypes.shape({
    current: PropTypes.shape({
      addFiles: PropTypes.func.isRequired,
    }),
  }).isRequired,
  fileList: PropTypes.array.isRequired,
  updateList: PropTypes.func.isRequired,
};

VideoList.propTypes = {
  classes: PropTypes.shape({
    root: PropTypes.string,
  }).isRequired,
  videos: PropTypes.array.isRequired,
  videoContent: PropTypes.arrayOf(PropTypes.string).isRequired,
  updateSelection: PropTypes.func.isRequired,
};

SearchPaper.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
  videos: PropTypes.array.isRequired,
  handleSearch: PropTypes.func.isRequired,
  handleGetVideoList: PropTypes.func.isRequired,
};

export default withStyles(styles, { withTheme: true })(SearchPaper);
