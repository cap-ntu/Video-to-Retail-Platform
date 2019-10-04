import React, { Component } from 'react';
import * as PropTypes from 'prop-types';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea/CardActionArea';
import CardMedia from '@material-ui/core/es/CardMedia/CardMedia';
import withStyles from '@material-ui/core/styles/withStyles';
import Typography from '@material-ui/core/es/Typography/Typography';
import Grid from '@material-ui/core/Grid/Grid';
import classNames from 'classnames';
import CardHeader from '@material-ui/core/CardHeader/CardHeader';
import Dialog from '@material-ui/core/Dialog';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogActions from '@material-ui/core/DialogActions';
import Route from 'react-router-dom/Route';
import Button from '../../common/Button';
import PlayOverlay from '../../common/overlay/PlayOverlay';
import CardMenu from '../../common/CardMenu';

const styles = theme => ({
  flat: {
    boxShadow: 'inherit',
    borderRadius: 0,
  },
  card: {
    display: 'inline-block',
  },
  cardPaper: {
    background: 'transparent',
  },
  imgCard: {
    width: 210,
    height: 118,
  },
  a: {
    textDecoration: 'none',
  },
  description: {
    padding: [[0, 18, 0, 6]],
  },
  moreButton: {
    transition: theme.transitions.create('opacity', {
      duration: theme.transitions.duration.shortest,
    }),
    '&:not(:hover)': {
      opacity: 0,
    },
    '&:hover': {
      backgroundColor: 'transparent',
    },
  },
  moreButtonShow: {
    opacity: [1, '!important'],
  },
  title: {
    width: 161,
  },
});

class VideoItemCard extends Component {
  state = {
    deleteDialog: false,
  };

  handleClick = history => {
    const { disable } = this.props;
    if (!disable) history.push(`/dashboard/watch?v=${this.props.id}`);
  };

  handleDelete = () => {
    this.props.handleDelete();
    this.setState({ deleteDialog: false });
  };

  render() {
    const {
      classes,
      cover,
      name,
      description,
      processed,
      overlay,
      disable,
      handleProcess,
    } = this.props;

    const { deleteDialog } = this.state;

    const menuItems = [];
    if (!processed) menuItems.push({ id: 'process', action: handleProcess });
    menuItems.push({
      id: (
        <Typography component='div' variant='body2' color='error'>
          delete
        </Typography>
      ),
      action: () => this.setState({ deleteDialog: true }),
    });

    return (
      <Card
        className={classNames(classes.card, classes.flat)}
        classes={{ root: classes.cardPaper }}
      >
        <Grid direction='column' container spacing={8}>
          <Grid item>
            <Card className={classes.card}>
              <Route
                render={({ history }) => (
                  <CardActionArea onClick={() => this.handleClick(history)}>
                    {!disable ? overlay : null}
                    <CardMedia
                      className={classes.imgCard}
                      component='img'
                      src={cover}
                      alt={name}
                    />
                  </CardActionArea>
                )}
              />
            </Card>
          </Grid>

          <Grid item>
            <Card
              className={classes.flat}
              style={{ overflow: 'visible' }}
              classes={{ root: classes.cardPaper }}
            >
              <CardHeader
                className={classes.description}
                action={
                  <CardMenu
                    menuItems={menuItems}
                    fontSize='small'
                    classes={{
                      button: classes.moreButton,
                      buttonMenuOpen: classes.moreButtonShow,
                    }}
                    disable={disable}
                  />
                }
                title={
                  <Typography
                    className={classes.title}
                    component='span'
                    variant='body1'
                    noWrap
                  >
                    {name}
                  </Typography>
                }
                subheader={
                  <Typography className={classes.title} variant='body2'>
                    {description}
                  </Typography>
                }
              />
            </Card>
          </Grid>
        </Grid>
        {/* delete dialog */}
        <Dialog
          classes={{ paper: classes.dialogPaper }}
          open={deleteDialog}
          onClose={() => this.setState({ deleteDialog: false })}
        >
          <DialogTitle>{`Delete video "${name}"?`}</DialogTitle>
          <DialogContent>
            <DialogContentText>
              {`Deleting model ${name} will also delete all its data. This action
              cannot be undo.`}
            </DialogContentText>
            <DialogActions>
              <Button
                color='primary'
                onClick={() => this.setState({ deleteDialog: false })}
              >
                Cancel
              </Button>
              <Button onClick={this.handleDelete}>
                <Typography color='error'>Delete Anyway</Typography>
              </Button>
            </DialogActions>
          </DialogContent>
        </Dialog>
      </Card>
    );
  }
}

VideoItemCard.defaultProps = {
  cover: `https://picsum.photos/210/118/?image=${Math.round(
    Math.random() * 200,
  )}`,
  overlay: <PlayOverlay />,
};

VideoItemCard.propTypes = {
  classes: PropTypes.object.isRequired,
  cover: PropTypes.string,
  name: PropTypes.string,
  description: PropTypes.string,
  id: PropTypes.string.isRequired,
  processed: PropTypes.bool.isRequired,
  overlay: PropTypes.node,
  disable: PropTypes.bool,
  handleProcess: PropTypes.func.isRequired,
  handleDelete: PropTypes.func.isRequired,
};

export default withStyles(styles)(VideoItemCard);
